"""
swish.py —— 编译并基准测试 swish.sycl 中的 6 个 SYCL/oneAPI kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 swish.sycl；
     PyTorch 会识别 .sycl 源文件并交给 icpx（DPC++）编译，同时链接 SYCL/XPU
     运行库（等价于 CUDA 版把 .cu 交给 nvcc）。
  2. 对多组 (S, K) 形状在 XPU 上生成随机张量（FP32 与 FP16）；
  3. run_benchmark() 先 warmup，再重复执行取平均耗时；
  4. 与脚本内自己拼的 torch_swish 对比正确性（打印前 2 个结果）和性能。

命名说明：swish 与 silu 是同一个函数，PyTorch 里的融合算子是
torch.nn.functional.silu；脚本里的 torch_swish 是 sigmoid + mul_ 两个算子拼出来的，
因此偏慢（不是公平对比）。

与 CUDA 版 swish.py 保持一致的一点：官方对照用脚本自拼的 torch_swish，
保留 out 参数原地写 y。

运行前需要 XPU 版 PyTorch（pip 安装源 https://download.pytorch.org/whl/xpu）
且 torch.xpu.is_available() 为 True。

用法：
  source ../../.venv/bin/activate
  python3 swish.py
"""

import os
import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load
import torch.utils.cpp_extension as ext

# TORCH_XPU_ARCH_LIST 未显式设置时置为空：让 PyTorch 只生成 spir64（JIT）设备代码，
# 避免对 intel/oneapi devel 镜像中缺失的 ocloc 做 AOT(spir64_gen) 编译。
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")

# swish 不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

if not torch.xpu.is_available():
    raise SystemExit(
        "torch.xpu 不可用：请确认已安装 XPU 版 PyTorch，并检查 Intel GPU 驱动。\n"
        "参考 scripts/README.md 执行环境搭建。"
    )

# Load the SYCL kernel as a python module
lib = load(
    name="swish_lib",
    sources=["swish.sycl"],
    # 常用编译选项：
    #   -O3                最高优化等级
    #   -std=c++20         由 PyTorch/SyclExtension 自动添加（torch 2.14 起要求 C++20）
    # 说明：CUDA 版的 TORCH_CUDA_ARCH_LIST 在这里对应 TORCH_XPU_ARCH_LIST；
    # 默认置空走 JIT(spir64)，PyTorch 自动添加 -fsycl-targets 等 SYCL 编译参数。
    # 对照：CUDA 版还用了 --use_fast_math（Swish 的开销主要在 exp 上，收益明显）；
    #      SYCL 侧默认用精度完整的 sycl::exp，如需对齐可换成 sycl::native::exp。
    extra_cflags=["-O3"],
    extra_sycl_cflags=["-O3"],
    verbose=False,
)

# 打印 PyTorch SYCL 扩展的实际构建目录（swish_lib.so 所在位置）
print(ext._get_build_directory("swish_lib", False))

# 打印当前 GPU 设备名（CUDA 版的 torch.cuda.get_device_name() 在 XPU 上的对应写法）
print(f"XPU device: {torch.xpu.get_device_name(0)}")


# 基准测试封装：
# - 传入 out 时走“原地写 y”的 kernel 接口（本模块 kernel 签名是 (x, y)）；
# - 不传 out 时函数返回新张量；
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
    # 不传 out 时 perf_func 会把结果作为返回值给出（对照实现走这条路径）
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    # 说明：CUDA 版还多了一行 [f"{v:<12}" for v in out_val]，会把数值转成字符串并
    # 左对齐补空格；LeetOneAPI 现有模块统一去掉了这一步，本模块沿用同样的格式。
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms Aha!")
    if show_all:
        print(out)
    return out, mean_time


# PyTorch 侧的对照片：脚本自己用 sigmoid + mul_ 两个算子拼出来的 swish。
# 数学上等价于 swish(x) = x * sigmoid(x)，但执行上是两个 elementwise kernel，
# 中间多读写一遍显存，所以耗时会明显高于单个融合 kernel
# （见 kernels/swish/README.md 里 out_f32_th 那一栏偏慢的原因）。
# 真正公平的对照应该是 torch.nn.functional.silu，本次保持脚本原样不替换。
def torch_swish(x, out=None):
    if out is None:
        return x * torch.sigmoid(x)
    else:
        torch.mul(x, torch.sigmoid(x), out=out)
        return out


# 测试规模：模拟类似矩阵 (S, K) 的形状，内部按 N = S * K 展平计算
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    # FP32 测试：连续内存布局是向量化访存的前提，所以显式 .contiguous()
    x = torch.randn((S, K)).xpu().float().contiguous()
    y = torch.zeros_like(x).xpu().float().contiguous()
    # 依次测：标量 / float4 向量化 / 对照实现
    run_benchmark(lib.swish_f32, x, "f32", y)
    run_benchmark(lib.swish_f32x4, x, "f32x4", y)
    run_benchmark(torch_swish, x, "f32_th", y)
    print("-" * 85)
    # FP32 尾数 23 位，最小刻度 2^-23 ≈ 0.00000012（约 7 位有效数字）；
    # FP16 尾数只有 10 位，最小刻度 2^-10 = 1/1024 ≈ 0.00098
    # （约 3~4 位有效数字，保守按约 3 位），FP32 转 half 会就近取整造成量化误差。
    # 另外 Swish 的 FP16 版本在 x <= -11.09 时会饱和成 -0.0（见 docs/kernels/swish/README.md）。
    # FP16 测试：从同一组随机数转成 half，便于对比精度损失
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    # 依次测：标量 / half2 / 8 元素 / 128 位打包 / 对照实现
    run_benchmark(lib.swish_f16, x_f16, "f16", y_f16)
    run_benchmark(lib.swish_f16x2, x_f16, "f16x2", y_f16)
    run_benchmark(lib.swish_f16x8, x_f16, "f16x8", y_f16)
    run_benchmark(lib.swish_f16x8_pack, x_f16, "f16x8pack", y_f16)
    run_benchmark(torch_swish, x_f16, "f16_th", y_f16)
    print("-" * 85)

print("swish benchmark done.")
