# Elementwise（逐元素加法）oneAPI / SYCL 移植设计说明

## 模块目标

本模块用于把 LeetCUDA 的 elementwise（逐元素加法）教学示例移植到 Intel oneAPI：

```text
c[i] = a[i] + b[i],  i = 0 .. N-1
```

在 CUDA 版中，同一个运算有 6 种写法，用于演示 **标量版 / 向量化 / FP16 / 128 位打包访存**
等带宽优化手段；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `elementwise_add_f32`：FP32 标量版（每 work-item 1 个元素）
2. `elementwise_add_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `elementwise_add_f16`：FP16 标量版（`sycl::half`）
4. `elementwise_add_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `elementwise_add_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存）
6. `elementwise_add_f16x8_pack`：FP16 每 work-item 8 个元素（16 字节打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> SYCL 没有独立的 `__global__` 函数，这些计算逻辑实际位于源码
> `submit_add_*` 函数里 `queue.submit + parallel_for` 提交的 kernel lambda 中；
> 下文为便于教学对照，仍按 6 个“kernel 版本”讲述。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及访存优化思想如何跨硬件迁移。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/elementwise/elementwise.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `elementwise.cu`，代码内含详细注释） |
| `kernels/elementwise/elementwise.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/elementwise/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid`（block 集合） | `nd_range<1>` 的 global range | 总 work-item 数 = grid × block |
| `block`（线程束集合） | nd_range 的 local range（work-group） | 每个 work-group 的 work-item 数 |
| 内置变量 `blockIdx.x` / `blockDim.x` / `threadIdx.x` | `item.get_group(0)` / `item.get_local_range(0)` / `item.get_local_id(0)` | work-item 在 nd_range 中的位置 |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<half,2>` | SIMD 向量类型 |
| `half` | `sycl::half` | FP16 类型，2 字节 |
| `__hadd` / `__hadd2` | 直接用 `sycl::half` 运算符 / 逐分量加法 | SYCL 支持 half 的 `+` 运算 |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构列表；本示例默认置空走 JIT(`spir64`)，避免 devel 镜像缺 `ocloc` 时 AOT 失败 |

## SYCL 线程组织回顾

一次 `parallel_for` 提交一个 **nd_range**，它由两部分组成：

```text
nd_range
├── global range（全部 work-item）
└── local range（work-group）
    └── work-item
```

进入 kernel lambda 后，每个 work-item 用 `nd_item` 上的接口确定自己的位置：

| 接口 | 含义 |
| --- | --- |
| `item.get_global_id(0)` | 当前 work-item 在整个 nd_range 中的全局编号 |
| `item.get_group(0)` | 当前 work-item 所在 work-group 的编号 |
| `item.get_local_id(0)` | 当前 work-item 在 work-group 内的编号 |
| `item.get_local_range(0)` | work-group 的大小（相当于 CUDA 的 blockDim） |

因此 CUDA 的经典写法

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

在 SYCL 中对应

```cpp
int idx = item.get_global_id(0);
```

例：`nd_range<1>(1024, 256)`、`N = 1000`：

- 总 work-item 数 = 1024，每个 work-group 256 个 work-item，共 4 个 group；
- group 0/1/2/3 分别处理下标 0~255、256~511、512~767、768~1023；
- 下标 1000~1023 的 work-item 被 `if (idx < N)` 拦截，不参与计算。

要点：

- 一次 `queue.submit` + `parallel_for` 只提交一次并行执行，与 CUDA “一次启动 = 一个 grid” 对应；
- `nd_range` 的 global range 必须是 local range 的整数倍；
- 同一 kernel lambda 可以被多次 submit，每次产生一次独立并行执行。

## 为什么 elementwise 适合并行

`c[i] = a[i] + b[i]` 中每个输出只依赖对应位置的输入，work-item 之间没有依赖、无需通信，
属于“尴尬并行”问题，适合切成很多份并行处理。

这类任务通常是**访存带宽瓶颈**，优化手段与 CUDA 完全一致：

- **访存合并/连续访问**：让同一时刻相邻的 work-item 访问相邻地址；
- **向量化**：每个 work-item 一次读写 4/8/16 字节，减少访存指令条数；
- **降低数据类型宽度**：FP16 只有 FP32 一半字节数，相同带宽可搬运两倍元素。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端展平为 `N = S * K`。

### 1. `elementwise_add_f32`（FP32 标量版）

```cpp
int idx = item.get_global_id(0);
if (idx < N) c[idx] = a[idx] + b[idx];
```

- 一个 work-item 只算一个元素；
- `if (idx < N)` 处理 `N` 不能被总 work-item 数整除时“多启动”的 work-item；
- 正确性最直观，是后续所有版本的基准。

### 2. `elementwise_add_f32x4`（FP32 向量化版）

每个 work-item 连续处理 **4 个 float（16 字节）**：

```cpp
int idx = 4 * item.get_global_id(0);
```

- 用 `reinterpret_cast` 把连续内存看成 `sycl::vec<float,4>`，一次加载/存储 16 字节；
- 必须用 `(idx + 3) < N` 判断整段不越界；
- 尾部不足 4 个的元素用标量循环补算，保证任意 `N` 都正确；
- 向量类型大小/对齐：4×4 字节 = 16 字节，torch 分配的 XPU 张量满足对齐要求。

### 3. `elementwise_add_f16`（FP16 标量版）

- 数据类型为 `sycl::half`（2 字节），字节数减半，是 FP16 向量化的基础；
- 精度代价与 CUDA 版相同：FP16 尾数只有 10 位，约 3~4 位十进制有效数字；
  FP32 转 half 会就近取整产生量化误差，属于预期精度损失。

### 4. `elementwise_add_f16x2`（FP16 每 work-item 2 元素）

- 把 `a[idx..idx+1]` 看成 `sycl::vec<half,2>`（2×2 字节 = 4 字节），一次完成两个元素的访存；
- 两个分量分别用 half 加法，然后一次写回。

### 5. `elementwise_add_f16x8`（FP16 每 work-item 8 元素）

- 每个 work-item 处理 8 个 half，通过 4 次 4 字节的向量读/写完成；
- 设计意图是比 f16x2 进一步摊薄索引计算与启动固定开销；
- 实测提醒：在 Arc A770 + JIT 下它反而可能比 f16x2 慢（见“性能观察”），
  说明“每线程多处理几个元素”不保证一定更快，硬件/编译器差异很大。

### 6. `elementwise_add_f16x8_pack`（128 位打包版）

- 8 个 half 正好 16 字节 = 128 位；
- 把 `a[idx..idx+7]` 看成 `sycl::vec<half,8>`，用一次 load/store 完成 16 字节搬运；
- 计算按 2 个 half 一组进行，与 CUDA 版“4 次 half2 操作”思路一致；
- 实测中它与 `f16x2` 是 FP16 版本里最快的两个，验证“更宽访存”在 Intel GPU 上确实有价值。

## 启动配置（host 端）

host 端与 CUDA 版保持同一套策略：

- 非二维张量展平成 `N`，每个 work-group 处理 256 个元素：

```cpp
const size_t local = 256 / n_elements;   // work-group 大小
const size_t grid  = (N + 255) / 256;    // work-group 数量
const size_t global = grid * local;      // nd_range 的 global range
```

- 二维输入 `(S, K)` 且满足以下三个条件时，才采用“一行一个 work-group”：

```text
1) 每行 work-item 数 K / n_elements > 0；
2) K / n_elements <= 1024；
3) K 能被 n_elements 整除（K % n_elements == 0）。
```

对应代码为：

```cpp
const size_t local  = K / n_elements;
const size_t global = S * local;
```

- 行过长或不可整除时，回退到上面的展平策略（与 `elementwise.sycl` 中
  `TORCH_BINDING_ELEM_ADD` 的实现保持一致）。

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（`torch.xpu`，实测版本 2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

PyTorch 对 Intel GPU 有两种设备代码生成方式：

### JIT（本示例默认）

`elementwise.py` 在 `load(...)` 前执行：

```python
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")
```

- 若用户没有显式设置 `TORCH_XPU_ARCH_LIST`，则将其置为空字符串；
- PyTorch 看到空列表后只生成 `-fsycl-targets=spir64`（通用格式）；
- kernel 运行时由 Intel GPU 驱动即时编译到具体 GPU；
- 不需要 `ocloc`，任何 Intel GPU 都能跑，缺点是首次运行有编译/加载开销。

### AOT

- 若不设置该变量，PyTorch 会回退到 `torch.xpu.get_arch_list()`，
  追加 `-fsycl-targets=spir64_gen,spir64` 并按 `pvc,bmg,arl-h,...` 等架构做离线编译；
- 离线生成具体 GPU 机器码需要 `ocloc` 工具，而本 devel 基础镜像没有安装，
  因此会报错：

  ```text
  ocloc tool could not be found and is required for AOT compilation
  ```

- 若机器已装好 `ocloc`，可自行指定：

  ```bash
  export TORCH_XPU_ARCH_LIST="你的架构名"
  python3 elementwise.py
  ```

> 另外注意：PyTorch 2.14 起要求 C++20，编译时不要用 `-std=c++17` 覆盖
> `SyclExtension` 自动添加的 `-std=c++20`。

## PyTorch 绑定

`elementwise.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python，
与 CUDA 版一一对应。主要工作：

1. 用宏检查张量 dtype 与 XPU 设备；
2. 根据维度/形状计算 nd_range；
3. 用 `data_ptr()` 拿裸指针（XPU 张量内存即 SYCL USM 内存，可直接交给 kernel）；
4. 从当前 XPU stream 取 `sycl::queue`，`queue.submit` 提交 kernel lambda。

调用链（以 `elementwise_add_f32` 为例）：

```text
elementwise.py
  └─ lib.elementwise_add_f32(a, b, c)              # Python 调用（pybind11）
      └─ elementwise_add_f32(...)                  # C++ host 包装函数（宏生成）
          └─ submit_add_f32(queue, ...)            # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range) # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

## 基准测试脚本

`elementwise.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `elementwise.sycl`（`.sycl` 源文件会被
   PyTorch 识别，交给 `icpx`/DPC++ 编译并链接 SYCL/XPU 运行库）；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合在 XPU 上生成随机张量；
3. `run_benchmark` 先 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与 PyTorch 官方 `torch.add` 的正确性和耗时。

## 性能观察与说明

### Arc A770 实测（S=4096, K=4096，JIT，一次运行结果）

| 版本 | 耗时 (ms) | 等效带宽 |
|---|---:|---:|
| f32 | 0.484 | ~416 GB/s |
| f32x4 | 0.486 | ~414 GB/s |
| torch f32 | 0.492 | ~409 GB/s |
| f16 | 0.300 | ~336 GB/s |
| f16x2 | 0.252 | ~399 GB/s |
| f16x8 | 0.376 | ~267 GB/s |
| f16x8_pack | 0.253 | ~397 GB/s |
| torch f16 | 0.254 | ~397 GB/s |

参考对照（同一张卡、同一 oneAPI 驱动下的临时实验）：

- 纯设备间 `memcpy`（显存到显存）：约 ~409 GB/s；
- 原生 SYCL 最小 add 基准：f32 约 ~416 GB/s、FP16 向量版约 ~397 GB/s；
- Arc A770 理论带宽 560 GB/s，当前 oneAPI/JIT 路径下实测约 71%~74%。

由此得到的结论：

1. FP16 最快版本约为 FP32 的 1.9 倍，符合“数据量减半、任务受带宽限制”的预期；
2. f32 标量、f32x4、torch f32 都停在 ~410~416 GB/s，说明 f32 已贴近
   当前环境的搬运上限，f32x4 不再有额外收益；
3. FP16 里 `f16x2` 与 `f16x8_pack` 最快，`f16x8` 反而明显偏慢；
   这与 CUDA/RTX 5070 Ti 上的表现不同——在 5070 Ti 上 `f16x8` 只比最优慢约 3%，
   说明**kernel 快慢与硬件/编译器强相关，不能把 CUDA 结论直接照搬到 oneAPI**；
4. 实测没有跑满 560 GB/s 理论值，但已接近“当前 oneAPI + Arc 驱动 + JIT + eGPU”
   这一组合能稳定提供的上限（纯 memcpy 也只有 ~409 GB/s）；
   是否还有余量，需要换 AOT/OpenCL 等路径进一步验证，不能简单归咎于 kernel。

### 跨硬件对比（5070 Ti，CUDA 参考）

| 指标 | Arc A770（oneAPI/JIT） | RTX 5070 Ti（CUDA） |
|---|---:|---:|
| 理论显存带宽 | 560 GB/s | 896 GB/s |
| FP32 实测最优 | ~416 GB/s（74%） | ~793 GB/s（89%） |
| FP16 实测最优 | ~399 GB/s（71%） | ~882 GB/s（98%） |
| f16x8 表现 | 明显偏慢 | 接近最优 |

对比意义：两张卡都能正确跑通全部 6 个版本；绝对耗时不能直接比，
应该看带宽利用率。5070 Ti 的 CUDA/驱动优化更成熟，而 Arc 当前结果
代表的是 Intel 这套软件栈的现实水平，不代表 oneAPI 的上限。

## 后续可以尝试的优化方向

- 安装含 `ocloc` 的工具链，对比 AOT 与 JIT 的带宽差距；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 使用 `sycl::ext::oneapi` 缓存/读取提示与 sub-group 技术；
- 在 Arc 上重点优化 `f16x8`，或直接用 `f16x2`/`f16x8_pack`；
- FP16 输入转 FP32 计算再转回，提升精度；
- 对比不同 work-group 大小、不同向量宽度对带宽的实测影响。
