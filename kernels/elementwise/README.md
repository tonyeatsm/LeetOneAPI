# Elementwise

## 0x00 说明

本模块是 LeetCUDA `kernels/elementwise` 的 Intel oneAPI/SYCL 移植版本，包含：

- [X] `elementwise_add_f32`：FP32 标量版（SYCL）
- [X] `elementwise_add_f32x4`：FP32 向量化版（`sycl::vec<float,4>`，16 字节访存）
- [X] `elementwise_add_f16`：FP16 标量版（`sycl::half`）
- [X] `elementwise_add_f16x2`：FP16 每 work-item 2 元素（4 字节访存）
- [X] `elementwise_add_f16x8`：FP16 每 work-item 8 元素（4 次 4 字节访存）
- [X] `elementwise_add_f16x8_pack`：FP16 每 work-item 8 元素（128 位打包访存）
- [X] PyTorch XPU bindings

代码文件：

| 文件 | 作用 |
| --- | --- |
| `elementwise.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `elementwise.cu`） |
| `elementwise.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |

## 测试

先按 `../../scripts/README.md` 完成环境搭建（Intel oneAPI 容器 + Python venv +
PyTorch XPU），然后运行：

```bash
# 进入 oneAPI 容器后手动执行（命令详见 ../../scripts/README.md）
cd /workspace/LeetOneAPI
source .venv/bin/activate
cd /workspace/LeetOneAPI/kernels/elementwise
python3 elementwise.py
```

默认 elementwise.py 使用 JIT（仅 spir64）编译，避免基础镜像缺少 ocloc 时 AOT 失败。
仅当已安装 ocloc/离线 GPU 编译器且需要 AOT 时，才自行设置：

```bash
export TORCH_XPU_ARCH_LIST="xe-hpg"  # 按实际架构填写，如 pvc、bmg 等
python3 elementwise.py
```

输出结构（数值取决于随机输入与具体 Intel GPU，不会与 CUDA 版逐位相同）：

```text
-------------------------------------------------------------------------------------
                                        S=1024, K=1024
           out_f32: [ ... ], time:0.00598025ms Aha!
         out_f32x4: [ ... ], time:0.00410318ms Aha!
        out_f32_th: [ ... ], time:0.00588393ms Aha!
-------------------------------------------------------------------------------------
           out_f16: [ ... ], time:0.00548601ms Aha!
         out_f16x2: [ ... ], time:0.00389791ms Aha!
         out_f16x8: [ ... ], time:0.00386930ms Aha!
     out_f16x8pack: [ ... ], time:0.00386310ms Aha!
        out_f16_th: [ ... ], time:0.00583792ms Aha!
-------------------------------------------------------------------------------------
...
```

判断标准：

- 同一次运行的 FP32 三个输出前 2 个元素应一致；
- 同一次运行的 FP16 各版本输出前 2 个元素应一致；
- FP16 与 FP32 允许有约 `1/1024` 量级的量化误差，这是半精度本身的精度损失；
- 通常向量化版本快于标量版本，FP16 快于 FP32，`f16x8_pack` 在多数规模下最快。

## CUDA → SYCL 速查

| CUDA | oneAPI / SYCL |
| --- | --- |
| `elementwise.cu` + nvcc | `elementwise.sycl` + `icpx -fsycl` |
| `kernel<<<grid, block>>>(args)` | `queue.submit` + `parallel_for(nd_range)` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `item.get_global_id(0)` |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<half,2>` |
| `torch.cuda` / `.cuda()` | `torch.xpu` / `.xpu()` |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` |

详细设计说明见 `../../docs/elementwise/README.md`。
