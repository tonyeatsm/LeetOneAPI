# Histogram

## 0x00 说明

本模块是 LeetCUDA `kernels/histogram` 的 Intel oneAPI/SYCL 移植版本，包含：

- [X] `histogram_i32`：int32 标量版（SYCL，每 work-item 1 个元素）
- [X] `histogram_i32x4`：int32 向量化版（`sycl::vec<int,4>`，16 字节访存）
- [X] PyTorch XPU bindings

代码文件：

| 文件 | 作用 |
| --- | --- |
| `histogram.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `histogram.cu`） |
| `histogram.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行简单测试 |

> 说明：本模块严格照搬 CUDA 版的算法与启动配置，**包括** `histogram_i32x4`
> 缺少尾部分支的已知缺陷（`N % 4 != 0` 时会越界读取）。测试数据长度 `10000`
> 是 4 的倍数，所以不会触发该问题。详见 `../../docs/kernels/histogram/README.md`。

## 测试

先按 `../../scripts/README.md` 完成环境搭建（Intel oneAPI 容器 + Python venv +
PyTorch XPU），然后运行：

```bash
# 进入 oneAPI 容器后手动执行（命令详见 ../../scripts/README.md）
cd /workspace/LeetOneAPI
source .venv/bin/activate
cd /workspace/LeetOneAPI/kernels/histogram
python3 histogram.py
```

默认 `histogram.py` 使用 JIT（仅 spir64）编译，避免基础镜像缺少 ocloc 时 AOT 失败。
仅当已安装 ocloc/离线 GPU 编译器且需要 AOT 时，才自行设置：

```bash
export TORCH_XPU_ARCH_LIST="xe-hpg"  # 按实际架构填写，如 pvc、bmg 等
python3 histogram.py
```

实测输出（Intel Arc A770，JIT/spir64）：

```text
/root/.cache/torch_extensions/py312_cpu/hist_lib
XPU device: Intel(R) Arc(TM) A770 Graphics
--------------------------------------------------------------------------------
h_i32   0: 1000
h_i32   1: 1000
h_i32   2: 1000
h_i32   3: 1000
h_i32   4: 1000
h_i32   5: 1000
h_i32   6: 1000
h_i32   7: 1000
h_i32   8: 1000
h_i32   9: 1000
--------------------------------------------------------------------------------
h_i32x4 0: 1000
h_i32x4 1: 1000
h_i32x4 2: 1000
h_i32x4 3: 1000
h_i32x4 4: 1000
h_i32x4 5: 1000
h_i32x4 6: 1000
h_i32x4 7: 1000
h_i32x4 8: 1000
h_i32x4 9: 1000
--------------------------------------------------------------------------------
```

判断标准：同一份 `[0..9] * 1000` 输入下，两个版本对 `0..9` 每个桶都应得到 `1000`。

## CUDA → SYCL 速查

| CUDA | oneAPI / SYCL |
| --- | --- |
| `histogram.cu` + nvcc | `histogram.sycl` + `icpx -fsycl` |
| `kernel<<<grid, block>>>(args)` | `queue.submit` + `parallel_for(nd_range)` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `item.get_global_id(0)` |
| `atomicAdd(&y[v], 1)` | `sycl::atomic_ref<int,...>(y[v]).fetch_add(1)` |
| `int4` | `sycl::vec<int,4>` |
| `torch.cuda` / `.cuda()` | `torch.xpu` / `.xpu()` |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` |

详细设计说明见 `../../docs/kernels/histogram/README.md`。
