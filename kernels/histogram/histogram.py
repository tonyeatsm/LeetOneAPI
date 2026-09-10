"""
histogram.py —— 编译并测试 histogram.sycl 中的 2 个 SYCL/oneAPI kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 histogram.sycl；
     PyTorch 会识别 .sycl 源文件并交给 icpx（DPC++）编译，同时链接 SYCL/XPU
     运行库（等价于 CUDA 版把 .cu 交给 nvcc）。
  2. 打印 PyTorch SYCL 扩展的实际构建目录（hist_lib.so 所在位置）；
  3. 构造 a = [0, 1, ..., 9] * 1000（长度 10000，元素均为非负整数）；
  4. 分别调用 histogram_i32（标量版）与 histogram_i32x4（int4 向量化版）；
  5. 打印每个桶的计数值，人工核对每个值是否都出现 1000 次。

运行前需要 XPU 版 PyTorch（pip 安装源 https://download.pytorch.org/whl/xpu）
且 torch.xpu.is_available() 为 True。建议指定目标 GPU 架构以缩短编译时间，
例如 export TORCH_XPU_ARCH_LIST="xe-hpg"（默认置空走 JIT(spir64)）。

用法：
  source ../../.venv/bin/activate
  python3 histogram.py
"""

import os

import torch
from torch.utils.cpp_extension import load

# 用于查询 PyTorch SYCL 扩展的实际构建目录
import torch.utils.cpp_extension as ext

# TORCH_XPU_ARCH_LIST 未显式设置时置为空：让 PyTorch 只生成 spir64（JIT）设备代码，
# 避免对 intel/oneapi devel 镜像中缺失的 ocloc 做 AOT(spir64_gen) 编译。
# 若机器已装好 ocloc 且希望 AOT，可自行 export TORCH_XPU_ARCH_LIST=...。
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")

# 直方图统计不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

if not torch.xpu.is_available():
    raise SystemExit(
        "torch.xpu 不可用：请确认已安装 XPU 版 PyTorch，并检查 Intel GPU 驱动。\n"
        "参考 scripts/README.md 执行环境搭建。"
    )

# 现场编译 histogram.sycl，得到一个 Python 可调用模块 lib
# Load the SYCL kernel as a python module
lib = load(
    name="hist_lib",
    sources=["histogram.sycl"],
    # 常用编译选项：
    #   -O3        最高优化等级
    #   -std=c++20 由 PyTorch/SyclExtension 自动添加（torch 2.14 起要求 C++20）
    # 说明：CUDA 版的 TORCH_CUDA_ARCH_LIST 在这里对应 TORCH_XPU_ARCH_LIST；
    # 默认置空走 JIT(spir64)，PyTorch 自动添加 -fsycl-targets 等 SYCL 编译参数。
    extra_cflags=["-O3"],
    extra_sycl_cflags=["-O3"],
    verbose=False,
)

# 打印 PyTorch SYCL 扩展的实际构建目录（hist_lib.so 所在位置）
print(ext._get_build_directory("hist_lib", False))

# 构造测试数据：0~9 每个值各出现 1000 次，总长度 N = 10 * 1000 = 10000。
# 元素都是非负 int32；且 N 恰好是 4 的倍数，因此向量化版本不会读到越界数据。
a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).xpu()

# 标量版：host 端用 max(a) + 1 = 10 决定桶的数量，一个 work-item 处理 1 个元素
h_i32 = lib.histogram_i32(a)
print("-" * 80)
for i in range(h_i32.shape[0]):
    print(f"h_i32   {i}: {h_i32[i]}")

# 向量化版：一个 work-item 处理 4 个元素（int4，16 字节访存），预期结果与标量版一致
# 注意：该版本与 CUDA 版一样缺少 tail 分支，仅当 N 为 4 的倍数时才安全（本例 N=10000）。
print("-" * 80)
h_i32x4 = lib.histogram_i32x4(a)
for i in range(h_i32x4.shape[0]):
    print(f"h_i32x4 {i}: {h_i32x4[i]}")
print("-" * 80)
