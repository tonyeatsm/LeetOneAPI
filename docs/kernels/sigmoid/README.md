# Sigmoid（S 型激活函数）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 sigmoid（S 型激活函数）教学示例移植到 Intel oneAPI：

```text
y[i] = 1 / (1 + exp(-x[i])),  i = 0 .. N-1
```

在 CUDA 版中，同一个运算有 6 种写法，用于演示 **标量版 / 向量化 / FP16 / 128 位打包访存**
等带宽与访存优化手段；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `sigmoid_f32`：FP32 标量版（每 work-item 1 个元素）
2. `sigmoid_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `sigmoid_f16`：FP16 标量版（`sycl::half`）
4. `sigmoid_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `sigmoid_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存，unpack 写法）
6. `sigmoid_f16x8_pack`：FP16 每 work-item 8 个元素（128 位打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> SYCL 没有独立的 `__global__` 函数，这些计算逻辑实际位于源码
> `submit_sigmoid_*` 函数里 `queue.submit + parallel_for` 提交的 kernel lambda 中；
> 下文为便于教学对照，仍按 6 个“kernel 版本”讲述。

与 elementwise 模块最关键的区别：**sigmoid 是单输入算子**。
加法要读 `a`、`b` 两份数据再写 `c` 一份，sigmoid 只读 `x` 一份数据写 `y` 一份，
访存量少一半，而每个元素都要算一次 `exp`，属于**计算偏重**的逐元素算子——
这一点直接决定了下面几个设计取舍（向量化收益有限、FP16 收益来自 exp 与访存双重减半）。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及数值稳定性（exp 溢出的 clamp 保护）、向量化、FP16 精度等
通用知识如何跨硬件迁移。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/sigmoid/sigmoid.cu` 的行为，**只做 CUDA →
SYCL 的等价移植，不改算法逻辑**，具体包括：

1. 复刻 6 个 kernel 的写法（标量 / `float4` / `half` / `half2` / unpack / pack）
   与 host 端启动配置；
2. 复刻 sigmoid 特有的 **exp 溢出保护**：先 `clamp` 再取指数，边界常数与 CUDA 版一致
   （FP32 `±88.3762626647949`；FP16 上界 `11.089866`、下界 `-9.704061`）；
3. **原样保留** CUDA 版 `f32x4` / `f16x2` / `f16x8` 缺少 tail 分支（只判断段首
   `(idx + 0) < N`）以及 `f16x8_pack` 尾部整段漏算的已知缺陷，仅在本文档与源码注释中
   照实标注（详见“当前边界约束”一节）；
4. 按已有 elementwise / histogram 模块的文档、脚本、代码组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/sigmoid/sigmoid.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `sigmoid.cu`，代码内含详细注释） |
| `kernels/sigmoid/sigmoid.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/sigmoid/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/kernels/sigmoid/README.md` | 本文件，模块设计说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid`（block 集合） | `nd_range<1>` 的 global range | 总 work-item 数 = grid × block |
| `block`（线程束集合） | nd_range 的 local range（work-group） | 每个 work-group 的 work-item 数 |
| 内置变量 `blockIdx.x` / `blockDim.x` / `threadIdx.x` | `item.get_group(0)` / `item.get_local_range(0)` / `item.get_local_id(0)` | work-item 在 nd_range 中的位置 |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<sycl::half,2>` | SIMD 向量类型 |
| `half` | `sycl::half` | FP16 类型，2 字节 |
| `expf` / `hexp` | `sycl::exp`（含 `sycl::half` 重载） | 指数函数 |
| `fminf` / `fmaxf` / `__hmin` / `__hmax` | `sycl::fmin` / `sycl::fmax`（对 `sycl::half` 也可用） | 夹取（clamp） |
| `__float2half(1.0f)` | `sycl::half(1.0f)` | 把常数 1.0 显式转成 half |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构列表；本示例默认置空走 JIT(`spir64`)，避免 devel 镜像缺 `ocloc` 时 AOT 失败 |

## sigmoid 曲线与数值性质

```text
  y = 1 / (1 + e^(-x))

1.0 ┤                     **********
    ┤                  ***
    ┤                **
0.5 ┤               +
    ┤             **
    ┤          ***
0.0 ┤**********
     └──────────────────────────────→ x
     -6      -3     0      3       6

采样值：-6→0.0025、-3→0.0474、0→0.5000、3→0.9526、6→0.9975
```

读图要点：

- S 形、单调递增：左边贴近 0、右边贴近 1，中间平滑过渡；
- 值域恒为 `(0, 1)` 开区间，输出可解释为概率 / 门控开关；
- 关于 `(0, 0.5)` 中心对称：`sigmoid(-x) = 1 - sigmoid(x)`；
  导数 `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))` 在 `x = 0` 处最大（0.25），
  所以中心附近最敏感、近似线性；
- **两端饱和**：`|x|` 超过约 8 之后输出几乎不再变化——这正是下面 kernel 先做
  `clamp`（截断）再算 `exp` 的依据：饱和后的钳位不改变结果，却能避免 `exp` 溢出；
- 作为激活函数：它给网络引入非线性（否则多层线性变换叠加仍是线性变换）；
  缺点是两端导数趋近 0，深网络容易梯度消失，现代隐藏层多用 ReLU，
  但二分类输出层、LSTM 门控等仍常见。

## 数值稳定性：为什么 sigmoid 要先 clamp 再取指数

sigmoid 要算 `exp(-x)`，而指数函数对输入范围很敏感：

- 参数太大 → `exp` 溢出成 `inf` → `1 / (1 + inf) = 0`，看起来“还行”，但已经丢掉了
  所有有效位；对 `x` 为负的大数（`exp(-x)` 参数为正且很大）就会走到这条路径；
- 参数太小 → `exp` 下溢成 `0` → `1 / (1 + 0) = 1`，同样只是饱和值；
- 真正的问题是**未定义的中间量**：不同精度下的可表示范围不同，一旦 `exp` 出
  `inf` 再做减法/除法就可能出现 `inf - inf = NaN`，结果直接坏掉。

因此 CUDA 版与 SYCL 版本都先把输入夹到“`exp` 一定不溢出”的区间：

```cpp
v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);   // CUDA: fminf/fmaxf
v = sycl::fmin(sycl::fmax(v, MIN_EXP_F32), MAX_EXP_F32);  // SYCL
y = 1.0f / (1.0f + expf(-v));
```

边界常数的来历：

| 精度 | 常数 | 数值 | 含义 |
| --- | --- | --- | --- |
| FP32 | `MAX_EXP_F32` / `MIN_EXP_F32` | `±88.3762626647949` | 略小于 `ln(FLT_MAX) ≈ 88.723`，是逐元素 sigmoid 常用的保守上界；`exp` 的参数落在 ±88.376 内一定是有限非零值 |
| FP16 | `MAX_EXP_F16` | `11.089866488461016` | `ln(65504)`，65504 是 half 的最大规格数 |
| FP16 | `MIN_EXP_F16` | `-9.704060527839234` | `ln(2^-14)`，`2^-14` 是 half 的最小规格正数 |

FP16 的两个边界**不对称**，是因为 half 能表示的正数范围本身就偏向 0 一侧
（上界 65504、下界 2^-14 ≈ 6.1e-5，量级差得很远）。

由于 sigmoid 在两端的饱和性质，`|x|` 超出这些边界后结果在对应精度下已经是 `1.0`
或 `0.0`，所以**钳位不改变最终可表示的结果**，只是把危险的中间量挡在 `exp` 之前。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端展平为 `N = S * K`。

### 1. `sigmoid_f32`（FP32 标量版）

```cpp
int idx = item.get_global_id(0);
if (idx < N) {
  float v = sycl::fmin(sycl::fmax(x[idx], MIN_EXP_F32), MAX_EXP_F32);
  y[idx] = 1.0f / (1.0f + sycl::exp(-v));
}
```

- 一个 work-item 只算一个元素；
- `if (idx < N)` 处理 `N` 不能被总 work-item 数整除时“多启动”的 work-item；
- 正确性最直观，是后续所有版本的基准。

### 2. `sigmoid_f32x4`（FP32 向量化版）

每个 work-item 连续处理 **4 个 float（16 字节）**：

```cpp
int idx = 4 * item.get_global_id(0);
float4_vec reg_x = LOAD_FLOAT4(x + idx);      // 一次读 16 字节
// 4 个分量分别 clamp、分别算 sigmoid
if ((idx + 0) < N) STORE_FLOAT4(y + idx, reg_y);   // 只判断段首（照搬 CUDA 版）
```

- 用 `reinterpret_cast` 把连续内存看成 `sycl::vec<float,4>`，一次加载/存储 16 字节；
- 与 elementwise 的 `f32x4` 不同，这里**只判断了 `(idx + 0) < N`**，没有 tail 分支，
  详见“当前边界约束”；
- 注意“向量化”省的只是访存指令：计算仍要按 4 个分量逐条展开，**每个分量都要做一次
  `exp`**，属于计算偏重的逐元素算子，因此向量化带来的加速比通常不如 elementwise 加法明显。

### 3. `sigmoid_f16`（FP16 标量版）

- 数据类型为 `sycl::half`（2 字节），字节数减半，是 FP16 向量化的基础；
- FP16 计算有两个必须注意的点（与 CUDA 版一致）：
  1. 常数 `1.0` 要显式转成 half（CUDA 用 `__float2half(1.0f)`，SYCL 用
     `sycl::half(1.0f)`），否则参与运算的类型不确定；
  2. 指数用 half 版本，夹取也用 half 版本，避免每次计算都插入多余的转换；
- 精度代价与 CUDA 版相同：FP16 尾数只有 10 位，约 3~4 位十进制有效数字；
  结果落在 `(0, 1)`，FP32 转 half 会就近取整产生量化误差，属于预期精度损失，
  差异体现在小数点后第 4 位左右。

### 4. `sigmoid_f16x2`（FP16 每 work-item 2 元素）

- 把 `x[idx..idx+1]` 看成 `sycl::vec<sycl::half,2>`（4 字节），一次完成两个元素的读取；
- 与 elementwise 的 `f16x2` 有一点不同：加法有现成的成对指令（CUDA `__hadd2`），
  而**指数没有成对版本**，所以两个分量仍要分别调用 `exp`；
- 同样只判断 `(idx + 0) < N`，没有 tail 分支。

### 5. `sigmoid_f16x8`（FP16 每 work-item 8 元素，unpack 写法）

- 每个 work-item 处理 8 个 half，通过 4 次 4 字节的向量读/写完成；
- 这里的 “unpack” 指：数据在内存里本就是连续的 8 个 half，本 kernel 不把它整体
  打包搬运，而是按 `vec<half,2>` 逐段读、逐段写；
- 设计意图是比 `f16x2` 进一步摊薄索引计算与启动固定开销；
- 8 个分量分别 clamp、分别算 sigmoid，写回时按段判断 `(idx + 0/2/4/6) < N`。

### 6. `sigmoid_f16x8_pack`（128 位打包版）

- 8 个 half 正好 16 字节 = 128 位；
- 把 `x[idx..idx+7]` 看成 `sycl::vec<sycl::half,8>`，用一次 load/store 完成 16 字节搬运，
  比 Kernel 5 的多条 4 字节访存指令更少；
- 计算部分仍是逐个 half 处理（`#pragma unroll` 展开 8 轮），与 CUDA 版
  “把局部数组整体当 128 位向量搬运、元素逐个算”的思路一致；
- 写回时只在 `(idx + 7) < N` 成立时执行，详见“当前边界约束”。

### 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

CUDA 版 `sigmoid.cu` 在几个向量化 kernel 上只保证了“段首不越界”，SYCL 移植版本
按既定目标**严格等价 CUDA 版行为**，因此以下缺陷原样保留，仅在此处与源码注释里照实标注：

| 版本 | 越界保护写法 | `N` 不是向量宽度整数倍时的后果 |
| --- | --- | --- |
| `sigmoid_f32x4` | `if ((idx + 0) < N)` | 末尾 work-item 会**越界写** `y[idx+1..idx+3]`，且最后不足 4 个元素没有 tail 回退 |
| `sigmoid_f16x2` | `if ((idx + 0) < N)` | 同上，越界写 `y[idx+1]` |
| `sigmoid_f16x8` | `if ((idx + 0/2/4/6) < N)` | `N` 落在 `idx+7` 时仍会写 `y[idx+7]`；尾部不足 8 个元素没有回退分支 |
| `sigmoid_f16x8_pack` | `if ((idx + 7) < N)` | 不会越界写，但**尾部不足 8 个元素的整段会被直接丢弃（漏算）** |
| `sigmoid_f32` / `sigmoid_f16` | `if (idx < N)` | 标量版没有该问题，任意 `N` 都正确 |

另外，`f32x4` / `f16x2` / `f16x8` / `f16x8_pack` 的**向量读取是写在判断之外的**，
即 `N` 不是向量宽度整数倍时还会多读最多一个向量的数据（CUDA 版同样如此）。

本模块基准测试的 `S`、`K` 都是 256 的倍数，`N = S * K` 一定是 256 的倍数，
上面这些情况都不会被触发，因此测试结果不受影响。

如需在生产中使用，应补充类似 elementwise 的 tail 分支：

```cpp
if ((idx + 3) < N) {
  // 整段 4 个元素都有效，走向量化路径
} else if (idx < N) {
  // 尾部不足 4 个元素，退回逐元素处理
  for (int i = 0; (idx + i) < N; ++i) { /* 逐元素 sigmoid */ }
}
```

并给 `f16x8_pack` 的“只写 `(idx + 7) < N`”补上逐元素回退分支。

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

- 行过长或不可整除时，回退到上面的展平策略。

> 与 CUDA 版的差异：CUDA 的 `TORCH_BINDING_SIGMOID` 只判断了 `(K / n_elements) <= 1024`，
> 没有 `K % n_elements == 0` 与 `> 0` 这两项检查。若 `K` 不能被向量宽度整除，
> CUDA 版会按“一行一个 block”启动，导致每行末尾几个元素永远不被处理（静默漏算）。
> SYCL 移植版本沿用 elementwise 模块的 3 项检查，这种情况回退到展平启动；
> 本模块基准的形状都能整除，两种写法的启动配置完全一致。

各版本的 work-group 大小（`local`）如下：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每个 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `sigmoid_f32` | 1 | 256 | 256 |
| `sigmoid_f32x4` | 4 | 64 | 256 |
| `sigmoid_f16` | 1 | 256 | 256 |
| `sigmoid_f16x2` | 2 | 128 | 256 |
| `sigmoid_f16x8` | 8 | 32 | 256 |
| `sigmoid_f16x8_pack` | 8 | 32 | 256 |

## PyTorch 绑定

`sigmoid.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python，
与 CUDA 版一一对应。主要工作：

1. 用宏检查 `x` / `y` 的 dtype 与 XPU 设备，并检查两者形状一致；
2. 根据维度/形状计算 nd_range 的 global / local；
3. 用 `data_ptr()` 拿裸指针（XPU 张量内存即 SYCL USM 内存，可直接交给 kernel）；
4. 从当前 XPU stream 取 `sycl::queue`，`queue.submit` 提交 kernel lambda。

调用链（以 `sigmoid_f32` 为例）：

```text
sigmoid.py
  └─ lib.sigmoid_f32(x, y)                    # Python 调用（pybind11）
      └─ sigmoid_f32(x, y)                    # C++ host 包装函数（宏生成）
          └─ submit_sigmoid_f32(queue, ...)   # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)   # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

与 elementwise 的绑定相比只少了 `b`/`c` 两个参数（单输入算子）：

| 模块 | host 函数签名 | kernel 参数 |
| --- | --- | --- |
| elementwise | `elementwise_add_xxx(a, b, c)` | `a, b, c, N` |
| sigmoid | `sigmoid_xxx(x, y)` | `x, y, N` |

## 基准测试脚本

`sigmoid.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `sigmoid.sycl`（`.sycl` 源文件会被
   PyTorch 识别，交给 `icpx`/DPC++ 编译并链接 SYCL/XPU 运行库）；
2. 打印构建目录与 XPU 设备名（与 elementwise / histogram 对齐）；
3. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合在 XPU 上生成随机张量
   （FP32 与 FP16 各一套，显式 `.contiguous()`）；
4. `run_benchmark` 先 warmup，再运行 1000 次取平均；
5. 依次对比 6 个自定义 kernel 与 PyTorch 官方 `torch.sigmoid` 的正确性和耗时。

与 CUDA 版 `sigmoid.py` 的对应关系：

| CUDA 版 | SYCL 版 | 说明 |
| --- | --- | --- |
| `torch.randn((S, K)).cuda()` | `torch.randn((S, K)).xpu()` | 设备张量创建 |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` | 计时前的同步 |
| `torch.cuda.get_device_name()` | `torch.xpu.get_device_name(0)` | 设备名 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST`（默认置空走 JIT） | 目标架构 |

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（`torch.xpu`，实测版本 2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

与 elementwise / histogram 相同，`sigmoid.py` 在 `load(...)` 前执行
`os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`：

- 未显式设置时只生成 `-fsycl-targets=spir64`（JIT），由 Intel GPU 驱动即时编译；
- 不需要 `ocloc`，任何 Intel GPU 都能跑，缺点是首次运行有编译/加载开销；
- 若机器已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 另外注意：PyTorch 2.14 起要求 C++20，编译时不要用 `-std=c++17` 覆盖
> `SyclExtension` 自动添加的 `-std=c++20`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 sigmoid.py`，结果为：

- FP32 三个版本（`sigmoid_f32` / `sigmoid_f32x4` / `torch.sigmoid`）在同一份输入上
  逐位一致；
- FP16 各版本（`sigmoid_f16` / `sigmoid_f16x2` / `sigmoid_f16x8` /
  `sigmoid_f16x8_pack`）彼此一致；`torch.sigmoid` 的 FP16 结果偶尔差 1 个 half ULP
  （约 `1e-4` 量级，例如 `0.47412109` vs `0.47436523`），属于 half 精度下的正常舍入差异；
- 所有结果都落在 `(0, 1)` 内，没有出现 `inf` / `NaN`，说明“先 clamp 再取指数”的
  溢出保护是有效的。

完整输出见 `kernels/sigmoid/README.md`。

### Arc A770 实测（S=4096, K=4096，JIT/spir64，一次运行结果）

sigmoid 是单输入算子：每个元素读 1 份、写 1 份，所以等效访存量 = `2 × N × sizeof(dtype)`，
与 elementwise（读 a、b 写 c，共 3 份）不同。

| 版本 | 耗时 (ms) | 等效带宽 |
| --- | ---: | ---: |
| f32 | 0.322 | ~417 GB/s |
| f32x4 | 0.326 | ~411 GB/s |
| torch f32 | 0.328 | ~409 GB/s |
| f16 | 0.289 | ~232 GB/s |
| f16x2 | 0.172 | ~391 GB/s |
| f16x8 | 0.326 | ~206 GB/s |
| f16x8_pack | 0.176 | ~381 GB/s |
| torch f16 | 0.182 | ~369 GB/s |

由此得到的结论：

1. **sigmoid 同样是访存带宽瓶颈**：FP32 三个版本都停在 ~410~417 GB/s，与 elementwise
   FP32 的实测上限（~416 GB/s）几乎相同；FP16 最快版本 ~391 GB/s，也与 elementwise
   FP16 的 ~399 GB/s 同量级。说明在这个数据规模下 `exp` 的计算开销被访存掩盖，
   耗时基本由“把数据搬两趟”决定；
2. FP16 最快版本比 FP32 快约 **1.9 倍**（0.322 ms → 0.172 ms），符合“字节数减半”的预期；
3. FP16 里 `f16x2` 与 `f16x8_pack` 最快，`f16x8`（unpack 写法）明显偏慢，已与 FP32
   持平——这与 elementwise 模块在 Arc A770 上的结论一致，说明“8 个元素拆成 4 次
   4 字节访存”在这套软件栈上不是好选择；
4. `f16` 标量版（~232 GB/s）明显慢于 `f16x2`，而 `f32` 标量版与 `f32x4` 基本持平：
   half 标量运算在这套栈上的相对开销更明显，FP16 更需要向量化；
5. 向量化对 FP32 没有收益（`f32x4` ≈ `f32` ≈ `torch f32`），与 elementwise 的
   `f32x4` 结论相同。

> 跨硬件对比：LeetCUDA 参考实现在同样 `S=4096, K=4096` 下 FP16 最快版本约 `0.021 ms`，
> 换算出的等效带宽超过该卡的理论显存带宽（约 896 GB/s），说明那组数字受 L2 缓存命中
> 或计时方式影响。因此跨硬件只比趋势（FP16 快于 FP32、向量/打包版快于 unpack 版），
> 不比绝对值。

## 后续可以尝试的优化方向

- 补上 `f32x4` / `f16x2` / `f16x8` 的 tail 分支与 `f16x8_pack` 的尾部回退，使任意
  `N` 都安全（对齐 LeetOneAPI elementwise 模块的写法）；
- 对比 `sycl::exp` 与 `sycl::native::exp`（快速近似版本，对应 CUDA 版
  `--use_fast_math` 的 `__expf`）的精度与速度差异——sigmoid 的开销主要在 `exp` 上，
  这一项通常比访存优化更有效；
- 试试用 `1 / (1 + exp(-x))` 的等价形式（如 `0.5 * (1 + tanh(0.5x))`）对比精度与性能；
- 安装含 `ocloc` 的工具链，对比 AOT 与 JIT 的差距；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 对比不同 work-group 大小、不同向量宽度对带宽/吞吐的实测影响。
