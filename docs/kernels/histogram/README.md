# Histogram（整数直方图统计）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 histogram（整数直方图统计）教学示例移植到 Intel oneAPI：

```text
y[v] = count(a[i] == v),  i = 0 .. N-1
```

输入 `a` 是一维 `int32` 张量，输出 `y` 是每个整数值出现的次数。

一句话理解：直方图就是「按值分堆数个数」——数值相同的元素看成放进了同一个桶，
最后统计每个桶里有多少个元素。

例如：

```text
a = [0, 1, 2, 0, 1, 0]
=> y[0] = 3, y[1] = 2, y[2] = 1
```

和 elementwise 不同的是，多个 work-item 可能同时命中同一个桶 `y[v]`，因此
**不能直接写** `y[a[idx]] += 1`，必须使用原子操作（CUDA 的 `atomicAdd`，
对应 SYCL 的 `sycl::atomic_ref`）保证结果正确。

CUDA 版包含 2 种写法，本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 2 种写法：

1. `histogram_i32`：int32 标量版（每 work-item 1 个元素）
2. `histogram_i32x4`：int32 向量化版（每 work-item 4 个元素，16 字节访存）

> 命名说明：上面 2 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> SYCL 没有独立的 `__global__` 函数，计算逻辑实际位于源码 `submit_hist_*` 函数里
> `queue.submit + parallel_for` 提交的 kernel lambda 中；下文为便于教学对照，
> 仍按 2 个“kernel 版本”讲述。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及「带写入冲突的并行」如何在 Intel GPU 上表达。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/histogram/histogram.cu` 的行为，**只做 CUDA →
SYCL 的等价移植，不改算法逻辑**，具体包括：

1. 复刻 2 个 kernel 的写法（标量版 / `int4` 向量化版）与 host 端启动配置；
2. 复刻「`max(a) + 1` 决定桶数量」的简化假设；
3. **原样保留** CUDA 版 `histogram_i32x4_kernel` 缺少 tail 分支的已知缺陷，
   仅在本文档与源码注释中照实标注（详见“当前边界约束”一节）；
4. 按 elementwise 模块的文档/脚本/代码组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/histogram/histogram.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `histogram.cu`，代码内含详细注释） |
| `kernels/histogram/histogram.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行简单测试 |
| `kernels/histogram/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/kernels/histogram/README.md` | 本文件，模块设计说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid`（block 集合） | `nd_range<1>` 的 global range | 总 work-item 数 = grid × block |
| `block`（线程束集合） | nd_range 的 local range（work-group） | 每个 work-group 的 work-item 数 |
| 内置变量 `blockIdx.x` / `blockDim.x` / `threadIdx.x` | `item.get_group(0)` / `item.get_local_range(0)` / `item.get_local_id(0)` | work-item 在 nd_range 中的位置 |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `int4` / `float4` | `sycl::vec<int,4>` / `sycl::vec<float,4>` | SIMD 向量类型 |
| `atomicAdd(&y[v], 1)` | `sycl::atomic_ref<int,...>(y[v]).fetch_add(1)` | 同一个桶的并发累加 |
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

## 为什么直方图需要原子操作

普通 elementwise 任务中每个输出位置只被一个 work-item 写，因此可以直接赋值：

```cpp
c[idx] = a[idx] + b[idx];
```

直方图的输出位置由**数据值**决定：

```cpp
int v = a[idx];   // 桶编号
y[v] += 1;        // 多个 work-item 可能同时写同一个 y[v]
```

如果两个 work-item 同时执行 `y[v] += 1`，会发生“读旧值、各自加 1、再写回”的竞争：

```text
work-item 0: read  y[v] = 0
work-item 1: read  y[v] = 0
work-item 0: write y[v] = 1
work-item 1: write y[v] = 1   // 本应得到 2，实际得到 1，丢了一次计数
```

SYCL 中把这个“读取 + 加 1 + 写回”做成一条不可分割的原子操作，写法是
`sycl::atomic_ref`：

```cpp
sycl::atomic_ref<int, sycl::memory_order::relaxed,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space>
    ref(y[v]);
ref.fetch_add(1);
```

- `memory_order::relaxed`：这里只要求“加 1 不丢”，不需要跨地址的先后顺序，
  是最轻量的序；
- `memory_scope::device`：保证同一个设备内所有 work-item 的原子操作互相可见
  （跨 work-group 的竞争也能正确累加）；
- `address_space::global_space`：`y` 是 device USM 全局内存指针。

这正是本模块与 elementwise 最本质的区别：
elementwise 是“没有数据依赖的尴尬并行”，histogram 是“带写入冲突的并行”。

## 本模块的简化假设

与 CUDA 版完全一致，当前实现默认输入满足：

- `a` 是一维 `int32` 张量；
- 所有元素均为**非负整数**；
- 直方图桶数由 `max(a) + 1` 决定（桶编号范围 `[0, M]`）。

如果输入含负数，`y[a[idx]]` 会越界，因此当前模块不处理负值输入。

## Kernel 设计

两个 kernel 都把输入输出当作一维数组处理，约定**每个 work-group 处理 256 个元素**。

### 1. `histogram_i32`（标量版）

```cpp
int idx = item.get_global_id(0);
if (idx < N)
  atomic_add(&y[a[idx]], 1);
```

- 一个 work-item 处理 1 个元素；
- `if (idx < N)` 处理 `N` 不能被总 work-item 数整除时“多启动”的 work-item；
- 每个有效 work-item 把自己的值 `a[idx]` 对应桶原子加 1。

### 2. `histogram_i32x4`（int4 向量化版）

每个 work-item 连续处理 4 个 `int32`（共 16 字节）：

```cpp
int idx = 4 * item.get_global_id(0);
if (idx < N) {
  sycl::vec<int,4> reg_a = LOAD_INT4(a + idx);
  atomic_add(&y[reg_a[0]], 1);
  atomic_add(&y[reg_a[1]], 1);
  atomic_add(&y[reg_a[2]], 1);
  atomic_add(&y[reg_a[3]], 1);
}
```

- 把 `a[idx..idx+3]` 的 16 字节 reinterpret 成 `sycl::vec<int,4>`，一次读入 4 个元素，
  减少访存指令条数；
- 四个分量分别做原子加 1；
- host 端对应启动：`local = 256 / 4 = 64` 个 work-item，每个 work-group 仍处理 256 个元素。

### 3. 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

`histogram_i32x4` 与 CUDA 版 `histogram_i32x4_kernel` 一样，**没有尾部分支**，
只判断了起点 `idx < N`：

```cpp
if (idx < N) {           // 只保证起点不越界
  ... LOAD_INT4(a + idx) // 当 N % 4 != 0 时，末尾线程可能多读 1~3 个元素
}
```

当 `N` 不是 4 的整数倍时，最后一个 work-item 的 `LOAD_INT4(a + idx)` 会越界读取
1~3 个元素。当前测试数据长度为 `10000`，正好是 4 的倍数，因此不会暴露该问题。

本次移植的既定目标是**严格等价 CUDA 版行为**，所以该缺陷原样保留，仅在本文档与
源码注释中照实标注，留待后续（对齐 LeetOneAPI elementwise 模块的 tail 写法）单独修复。

如需在生产中使用，应补充类似 elementwise 的 tail 分支：

```cpp
if ((idx + 3) < N) {
  // 4 个元素都有效，走 int4
} else if (idx < N) {
  // 尾部不足 4 个元素，退回逐元素处理
  for (int i = 0; (idx + i) < N; ++i)
    atomic_add(&y[a[idx + i]], 1);
}
```

## 启动配置（host 端）

host 端与 CUDA 版保持同一套策略。先通过 `torch::max(a, 0)` 得到输入最大值 `M`，
再创建长度 `M + 1` 的 `int32` 输出张量，桶编号范围是 `[0, M]`：

```cpp
std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);
const int M = std::get<0>(max_a).cpu().item().to<int>();
auto y = torch::zeros({M + 1}, options);   // options: int32 + XPU
```

随后按每个 work-group 处理 256 个元素计算启动配置：

```cpp
const size_t local_range = 256 / n_elements;                       // work-group 大小
const size_t num_groups  = (N + 256 - 1) / 256;                    // work-group 数量
const size_t global_range = num_groups * local_range;              // nd_range 的 global
```

所以：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `histogram_i32` | 1 | 256 | 256 |
| `histogram_i32x4` | 4 | 64 | 256 |

## PyTorch 绑定

`histogram.sycl` 通过 `TORCH_BINDING_HIST` 宏生成 2 个 host 函数，再经
`PYBIND11_MODULE` 暴露给 Python：

```text
histogram.py
  └─ lib.histogram_i32(a)                        # Python 调用（pybind11）
      └─ histogram_i32(torch::Tensor a)          # C++ host 包装函数（宏生成）
          ├─ torch::max(a, 0)                    # 求最大值，确定桶数量
          ├─ torch::zeros({M + 1}, int32 XPU)    # 创建输出桶数组
          ├─ 计算 nd_range 的 global / local
          └─ submit_hist_i32(queue, ...)         # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)
                  └─ Intel GPU 各 work-item 原子加到对应桶
```

host 函数的主要工作：

1. `CHECK_TORCH_TENSOR_DTYPE` 检查输入为 `torch::kInt32`，并检查张量在 XPU 上；
2. 用 `torch::max` 得到最大值 `M`；
3. 创建 `M + 1` 个桶；
4. 计算 `global / local`（`nd_range`）；
5. `data_ptr()` 取裸指针（XPU 张量内存即 SYCL USM 内存，可直接交给 kernel），
   从当前 XPU stream 取 `sycl::queue` 并提交 kernel。

## 测试脚本

`histogram.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `histogram.sycl`
   （`.sycl` 源文件会被 PyTorch 识别，交给 `icpx`/DPC++ 编译并链接 SYCL/XPU 运行库）；
2. 用 `torch.utils.cpp_extension._get_build_directory("hist_lib", False)` 打印
   PyTorch SYCL 扩展的实际构建目录（`hist_lib.so` 所在位置），并用
   `torch.xpu.get_device_name(0)` 打印 XPU 设备名（与 `elementwise.py` 对齐）；
3. 生成 `a = [0,1,...,9] * 1000`，长度 `10000`（非负 int32，且是 4 的倍数）；
4. 分别调用 `histogram_i32` 和 `histogram_i32x4`；
5. 打印每个桶的计数值。

预期每个值出现 `1000` 次：

```text
h_i32   0: 1000
h_i32   1: 1000
...
h_i32   9: 1000
h_i32x4 0: 1000
h_i32x4 1: 1000
...
h_i32x4 9: 1000
```

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（`torch.xpu`）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

与 elementwise 模块相同，`histogram.py` 在 `load(...)` 前执行
`os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`：

- 未显式设置时只生成 `-fsycl-targets=spir64`（JIT），由 Intel GPU 驱动即时编译；
- 不需要 `ocloc`，任何 Intel GPU 都能跑，缺点是首次运行有编译/加载开销；
- 若机器已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 另外注意：PyTorch 2.14 起要求 C++20，编译时不要用 `-std=c++17` 覆盖
> `SyclExtension` 自动添加的 `-std=c++20`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 histogram.py`，2 个版本都对同一份
`[0..9] * 1000` 输入得到 `0..9` 每个桶恰好 `1000`，与 CUDA 版行为一致。
详细输出见 `kernels/histogram/README.md`。

值得注意的点：

1. 直方图比 elementwise 更容易出现原子竞争：本例只有 10 个桶，相当于 10000 次
   原子操作集中打到 10 个地址上，热点桶的竞争是主要瓶颈；
2. `i32x4` 减少的是访存指令条数（一次 16 字节读入 4 个元素），但 4 次原子加
   之间的桶冲突依然存在，因此向量化对直方图的收益通常不如 elementwise 明显；
3. 本例只用于验证正确性，`histogram.py` 不含计时逻辑；若要比较性能，
   应改用更大桶数、更分散的输入分布，并按 elementwise 的方式加 warmup + 多次取平均。

## 后续可以尝试的优化方向

- 补上 `histogram_i32x4` 的尾部分支，使任意 `N` 都安全（对齐 elementwise 模块）；
- 使用 work-group 局部（local memory）直方图，最后再合并到全局，减少原子冲突；
- 使用 sub-group（如 `sycl::ext::oneapi` 的 ballot / group 归约）降低竞争；
- 支持更大桶数或负值索引（例如统一加偏移让索引非负）；
- 对比不同 work-group 大小、不同向量宽度对直方图吞吐的实测影响。
