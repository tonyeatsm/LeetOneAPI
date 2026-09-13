"""
gelu.py —— 编译并基准测试 gelu.sycl 中的 6 个 SYCL/oneAPI kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 gelu.sycl；
     PyTorch 会识别 .sycl 源文件并交给 icpx（DPC++）编译，同时链接 SYCL/XPU
     运行库（等价于 CUDA 版把 .cu 交给 nvcc）。
  2. 对多组 (S, K) 形状在 XPU 上生成随机张量（FP32 与 FP16）；
  3. run_benchmark() 先 warmup，再重复执行取平均耗时；
  4. 与 PyTorch 官方 tanh 近似 GELU（torch.nn.GELU("tanh")）对比正确性和性能。

注意：GELU 的 tanh 近似在 FP16 下用 exp 拼实现，存在相减抵消的精度损失；
当输入 x ≳ 4.03 时 exp 还会溢出成 inf、算出 NaN（详见
docs/kernels/gelu/README.md 与 kernels/gelu/README.md 的说明）。

与 CUDA 版 gelu.py 保持一致的一点：官方对照用 partial(torch.nn.GELU("tanh"))，
该模块实例没有 out 参数，所以走 run_benchmark 的“不传 out”分支。
必须显式写 "tanh"——PyTorch 默认是 approximate='none'（erf 精确式），
与 kernel 的 tanh 近似会有可见差异。

运行前需要 XPU 版 PyTorch（pip 安装源 https://download.pytorch.org/whl/xpu）
且 torch.xpu.is_available() 为 True。

用法：
  source ../../.venv/bin/activate
  python3 gelu.py
"""

import os
import time
from functools import partial
from typing import Optional

import torch
import torch.nn
import torch.utils
from torch.utils.cpp_extension import load
import torch.utils.cpp_extension as ext

# TORCH_XPU_ARCH_LIST 未显式设置时置为空：让 PyTorch 只生成 spir64（JIT）设备代码，
# 避免对 intel/oneapi devel 镜像中缺失的 ocloc 做 AOT(spir64_gen) 编译。
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")

# gelu 不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

if not torch.xpu.is_available():
    raise SystemExit(
        "torch.xpu 不可用：请确认已安装 XPU 版 PyTorch，并检查 Intel GPU 驱动。\n"
        "参考 scripts/README.md 执行环境搭建。"
    )

# Load the SYCL kernel as a python module
lib = load(
    name="gelu_lib",
    sources=["gelu.sycl"],
    # 常用编译选项：
    #   -O3                最高优化等级
    #   -std=c++20         由 PyTorch/SyclExtension 自动添加（torch 2.14 起要求 C++20）
    # 说明：CUDA 版的 TORCH_CUDA_ARCH_LIST 在这里对应 TORCH_XPU_ARCH_LIST；
    # 默认置空走 JIT(spir64)，PyTorch 自动添加 -fsycl-targets 等 SYCL 编译参数。
    # 对照：CUDA 版还用了 --use_fast_math（GELU 的开销主要在 tanh 上，收益明显；
    #      但它也让 half 版的溢出行为更“干脆”，x ≳ 4.03 时直接得到 NaN）；
    #      SYCL 侧默认用精度完整的 sycl::tanh / sycl::exp。
    extra_cflags=["-O3"],
    extra_sycl_cflags=["-O3"],
    verbose=False,
)

# 打印 PyTorch SYCL 扩展的实际构建目录（gelu_lib.so 所在位置）
print(ext._get_build_directory("gelu_lib", False))

# 打印当前 GPU 设备名（CUDA 版的 torch.cuda.get_device_name() 在 XPU 上的对应写法）
print(f"XPU device: {torch.xpu.get_device_name(0)}")


# 基准测试封装：
# - 传入 out 时走“原地写 y”的 kernel 接口（本模块 kernel 签名是 (x, y)）；
# - 不传 out 时函数返回新张量（torch.nn.GELU 是模块实例，没有 out 参数）；
# - 先 warmup 让 GPU 时钟与缓存达到稳定状态，再正式计时取平均。
def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # 清空输出张量，避免上一次测试的残留数据影响结果
    if out is not None:
        out.fill_(0)
    # warmup：先跑若干次，让 kernel 加载、GPU 频率和缓存达到稳定状态。
    # 对照实现（nn.Module）没有 out 参数，因此这里区分了两条调用路径。
    if out is not None:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    # 同步：确保前面所有 XPU 任务执行完，CPU 计时点才是准确的
    torch.xpu.synchronize()

    start = time.time()
    # 正式计时：连续执行 iters 次，最后取平均单次耗时
    if out is not None:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)
    torch.xpu.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # 单位换算为毫秒
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    # XPU 张量 → 拉平 → 断开可能的梯度追踪（本脚本已关 autograd，属保险写法）
    # → 拷回 CPU → 转 numpy → 转 Python 列表，再取前 2 个元素
    # 只打印前 2 个元素用于人工核对正确性
    # （CUDA 版 gelu.py 本来就没有 f"{v:<12}" 的左对齐补齐，输出是干净的
    #   [-0.13358943, -0.06881647] 形式，LeetOneAPI 保持一致）
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms Aha!")
    if show_all:
        print(out)
    return out, mean_time


# 测试规模：模拟类似矩阵 (S, K) 的形状，内部按 N = S * K 展平计算
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]
# PyTorch 的对照片：torch.nn.GELU("tanh") 是一个 nn.Module **实例**，
# 必须显式写 "tanh" 才能和本模块的 tanh 近似对齐
# （默认是 approximate='none'，即 erf 精确式，两者会有可见差异）。
# 把模块实例赋给 torch.gelu 这个名字是脚本里的临时写法（照搬 CUDA 版），
# 下面再用 partial(...) 包一层，让调用形式与其它模块保持一致。
# 该模块实例没有 out 参数，所以对照测试走 run_benchmark 的“不传 out”分支。
torch.gelu = torch.nn.GELU("tanh")
for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    # FP32 测试：连续内存布局是向量化访存的前提，所以显式 .contiguous()
    x = torch.randn((S, K)).xpu().float().contiguous()
    y = torch.zeros_like(x).xpu().float().contiguous()
    # 依次测：标量 / float4 向量化 / 对照实现
    run_benchmark(lib.gelu_f32, x, "f32", y)
    run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
    run_benchmark(partial(torch.gelu), x, "f32_th")
    print("-" * 85)
    # FP16 测试：从同一组随机数转成 half。
    # 量化误差 + exp 拼 tanh 的相减抵消都会体现在这一组里；
    # 另外 torch.randn 在大尺寸下会出现个别 |x| > 4.03 的元素，
    # 它们会让本模块的 FP16 结果变成 NaN（抽查前 2 个元素看不出来）。
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    # 依次测：标量 / half2 / 8 元素 / 128 位打包 / 对照实现
    run_benchmark(lib.gelu_f16, x_f16, "f16", y_f16)
    run_benchmark(lib.gelu_f16x2, x_f16, "f16x2", y_f16)
    run_benchmark(lib.gelu_f16x8, x_f16, "f16x8", y_f16)
    run_benchmark(lib.gelu_f16x8_pack, x_f16, "f16x8pack", y_f16)
    run_benchmark(partial(torch.gelu), x_f16, "f16_th")
    print("-" * 85)

print("gelu benchmark done.")
