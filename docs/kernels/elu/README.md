# ELU（Exponential Linear Unit，指数线性单元）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 elu（指数线性单元）教学示例移植到 Intel oneAPI：

```text
elu(x) = { x,                    x > 0
         { alpha * (exp(x) - 1), x <= 0        （本模块 alpha = 1.0，见 ALPHA 宏）

y[i] = elu(x[i]),  i = 0 .. N-1
```

在 CUDA 版中，同一个运算有 6 种写法，用于演示 **标量版 / 向量化 / FP16 / 128 位打包访存**
等带宽与访存优化手段；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `elu_f32`：FP32 标量版（每 work-item 1 个元素）
2. `elu_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `elu_f16`：FP16 标量版（`sycl::half`）
4. `elu_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `elu_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存，unpack 写法）
6. `elu_f16x8_pack`：FP16 每 work-item 8 个元素（128 位打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> SYCL 没有独立的 `__global__` 函数，这些计算逻辑实际位于源码 `submit_elu_*`
> 函数里 `queue.submit + parallel_for` 提交的 kernel lambda 中；下文为便于教学对照，
> 仍按 6 个“kernel 版本”讲述。

**与 elementwise / sigmoid / relu 的关系**：本模块与它们是一组姊妹示例，6 个版本一一对应，
区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` / `y[i] = max(0, x[i])`
换成 `y[i] = elu(x[i])`。work-item 编号、向量化访存、FP16 打包等知识不再重复。

**本模块的独特价值**：ELU 是“把 ReLU 和 sigmoid 各取一半”的算子——正半轴和 ReLU 一样
是恒等映射（不饱和、梯度恒为 1），负半轴像 sigmoid 一样指数饱和到 `-alpha`，
但**指数只出现在负半轴**。于是它同时给出两个对照结论：

- 性能上介于 ReLU（纯访存瓶颈）与 sigmoid（计算偏重）之间，是观察
  “计算量增加多少才会打破访存瓶颈”的好样本；
- 数值上**不需要** sigmoid / gelu 那样的 `clamp` 溢出保护，
  因为大正数走恒等分支、大负数取 exp 只会下溢到 0（恰好是数学极限）。

与同族算子的对照：

| 对比项 | sigmoid | relu | elu（alpha = 1） |
| --- | --- | --- | --- |
| 公式 | `1/(1+exp(-x))` | `max(0, x)` | `x > 0 ? x : exp(x) - 1` |
| 值域 | `(0, 1)`，两端饱和 | `[0, +∞)`，仅负端截断 | `(-1, +∞)`，仅负端饱和 |
| `x = 0` 处 | 可导，导数 0.25 | 不可导（左 0 右 1） | 可导，导数 1 |
| 负半轴梯度 | 趋近 0（饱和） | 恒为 0（死亡 ReLU） | `exp(x) > 0`（衰减但不为 0） |
| 每元素计算 | 一次 `exp` + 一次加法 + 一次除法 | 一次取最大值 | 一次比较 + 一次 `exp`（仅负半轴） |
| 是否需要 clamp | 需要 | 不需要 | 不需要（见下节） |
| 成对指令 | 无 `hexp2`，只能逐分量算 | 有 `__hmax2`，可一次算 2 个 half | 同 sigmoid，无成对指令 |

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及溢出的“位置”（分母 / 分子 / 自己除自己）如何决定要不要做保护。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/elu/elu.cu` 的行为，**只做 CUDA → SYCL
的等价移植，不改算法逻辑**，具体包括：

1. 复刻 6 个 kernel 的写法（标量 / `float4` / `half` / `half2` / unpack / pack）、
   `ALPHA = 1.0f` 以及 host 端启动配置；
2. **不做** clamp / 溢出保护——CUDA 版没有 `MAX_EXP_*` 宏，SYCL 版同样不加
   （原因见“为什么 ELU 不需要溢出保护”一节）；
3. 复刻 ELU 特有的两个 `__device__` 辅助函数：FP32 的 `elu` 与 FP16 的 `elu_half`
   （SYCL 侧对应 `elu` 与 `elu_half` 两个 `static inline` 函数）；
4. **原样保留** CUDA 版 `f32x4` / `f16x2` / `f16x8` 缺少 tail 分支
   （只判断段首 `(idx + 0) < N`）以及 `f16x8_pack` 尾部整段漏算的已知缺陷，
   仅在本文档与源码注释中照实标注；
5. 测试脚本的官方对照**照搬 CUDA 版**：使用脚本自己用
   `gt / exp / sub / mul / where` 拼出来的 `torch_elu`（alpha = 1）。
   它数学上等价于 `torch.nn.functional.elu`，但由多个 elementwise kernel 组成、
   反复读写显存，所以 `out_*_th` 那一栏明显偏慢——这是 CUDA 版原本的口径，
   本次保持脚本原样不替换；
6. 打印格式沿用 LeetOneAPI 现有模块：`[-0.44692516, 0.08315884], time:0.04539156ms Aha!`，
   不再使用 CUDA 版 `f"{v:<12}"` 的引号与左对齐补空格；
7. 按已有 elementwise / histogram / relu / sigmoid 模块的组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/elu/elu.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `elu.cu`，代码内含详细注释） |
| `kernels/elu/elu.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/elu/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/kernels/elu/README.md` | 本文件，模块设计说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid` / `block` | `nd_range<1>` 的 global / local range | 总 work-item 数 = grid × block |
| `blockIdx.x` / `blockDim.x` / `threadIdx.x` | `item.get_group(0)` / `item.get_local_range(0)` / `item.get_local_id(0)` | work-item 在 nd_range 中的位置 |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<sycl::half,2>` | SIMD 向量类型 |
| `half` | `sycl::half` | FP16 类型，2 字节 |
| `expf` / `hexp` | `sycl::exp`（含 `sycl::half` 重载） | 指数函数 |
| `__hgt(x, 0)` | `x > sycl::half(0.0f)` | half 的“大于”比较（返回 bool） |
| `__hmul` / `__hsub` / `__hadd` / `__hdiv` / `__hneg` | `sycl::half` 的 `*` / `-` / `+` / `/` 与一元 `-` | half 的四则运算，SYCL 已定义运算符重载 |
| `__float2half(1.0f)` | `sycl::half(1.0f)` | 把常数显式转成 half |
| 三元表达式 `cond ? a : b` | 相同的三元表达式 | 逐元素选择，编译器会做谓词化 |
| `LDST128BITS(value)`（`reinterpret_cast<float4*>`） | `sycl::vec<sycl::half,8>` 的 load/store 宏 | 128 位整体访存 |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构；本示例默认置空走 JIT(`spir64`) |

## ELU 曲线与数学性质

```text
  y = x (x > 0)，y = alpha * (exp(x) - 1) (x <= 0)

3.0 ┤                          ***
    ┤                       ***
2.0 ┤                    ***
    ┤                 ***
1.0 ┤              ***
    ┤           ***
0.0 ┤**********+
    ┤      ****
-0.5┤  ****
-1.0┤*····························  ← 渐近线 y = -alpha = -1
    └──────────────────────────────→ x
    -3   -2   -1    0    1    2    3

采样值：-3→-0.9502、-2→-0.8647、-1→-0.6321、0→0、1→1、2→2、3→3
```

读图要点：

- **正半轴是恒等映射**：`x > 0` 时 `elu(x) = x`，这一半与 ReLU 完全一致，
  不饱和、梯度恒为 1；
- **负半轴是指数饱和**：`x <= 0` 时 `elu(x) = alpha * (exp(x) - 1)`，
  单调递增、有界，`x → -∞` 时极限是 `-alpha`（本模块即 -1），
  但**永远取不到** `-alpha`，所以值域是开区间 `(-alpha, +∞)`；
- **在 `x = 0` 处连续且可导**：左导数 `exp(0) = 1`、右导数 `1`，两侧相等，
  因此 `alpha = 1` 时 ELU 在原点一阶可导（补上了 ReLU 在 0 点不可导的短板）；
  `elu(0)` 走的是 `else` 分支，算出来是 `exp(0) - 1 = 0`，与正半轴严格衔接；
- **负半轴导数不为 0**：`elu'(x) = exp(x) ∈ (0, 1]`，与 ReLU 的“负半轴梯度恒为 0”
  形成关键对比——ELU 不会出现**死亡 ReLU（dead ReLU）**；
- **输出均值更接近 0**：负半轴输出负值而不是硬零，缓解了 ReLU 输出恒非负、
  把下一层偏置整体推偏的问题；
- **代价**：负半轴要算一次指数，比 ReLU 贵；因此本模块的性能特征介于
  ReLU（纯访存瓶颈）与 sigmoid（计算偏重）之间。

## 为什么 ELU 不需要溢出保护（与 sigmoid / gelu 的关键差异）

sigmoid 必须先 `clamp` 再 `exp`，gelu 也必须先 `clamp`（它的 tanh 是用 exp 拼的，
`(exp(2t)-1)/(exp(2t)+1)` 会 `inf/inf = NaN`）。ELU 不需要，原因在于
**指数只出现在负半轴**：

| 输入情况 | `elu` 的结果 |
| --- | --- |
| `x` 是很大的正数（如 `3.4e38`） | 走 `x > 0` 分支，原样返回，不产生新数值 |
| `x` 是很大的负数（如 `-3.4e38`） | `exp(x)` 下溢成 0，结果是 `alpha * (0 - 1) = -alpha`，正好是极限值 |
| `x = 0` | `exp(0) - 1 = 0`，与正半轴严格连续 |
| `x = +inf` / `-inf` | `+inf` / `-alpha` |

即使编译器把三元表达式编译成“两条分支都算、再按谓词选择”（常见的谓词化写法），
正半轴上算出的 `exp(大正数) = inf` 也只会被丢掉，不会污染结果——因为此时选择的是 `x` 本身。

**NaN 语义**（与 relu / gelu 对照时值得注意）：分支条件是 `x > 0`，
NaN 的比较结果为假，于是走进 `exp(NaN) - 1`，**NaN 被正常传播到输出**，
与 PyTorch 的 `torch.nn.functional.elu` 行为一致；而 relu 用的 `fmax(NaN, 0) = 0`
会把 NaN 吞掉，gelu 的 `fmax`/`fmin` 夹取同理。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 0. 两个辅助函数：`elu` 与 `elu_half`

CUDA 版把公式封装成两个 `__device__ __forceinline__` 函数（强制内联，
逐元素算子每个 work-item 都要调用一次）；SYCL 侧对应两个 `static inline` 函数，
`static` 让它们只在本翻译单元可见，`inline` 语义上与 `__forceinline__` 一致，
都交给编译器决定内联：

```cpp
// FP32：三元选择，一条比较 + 两条分支 + 一次选择
static inline float elu(float x) {
  return x > 0.0f ? x : ALPHA * (sycl::exp(x) - 1.0f);
}

// FP16：half 没有隐式类型提升，常数要用 sycl::half(...) 显式转换；
//       CUDA 的 __hgt / __hmul / __hsub / hexp 在 SYCL 里对应
//       比较运算符、* / - 与 sycl::exp
static inline sycl::half elu_half(sycl::half x) {
  return x > sycl::half(0.0f)
             ? x
             : sycl::half(ALPHA) * (sycl::exp(x) - sycl::half(1.0f));
}
```

命名保持 CUDA 版原样（FP32 直接叫 `elu`，FP16 叫 `elu_half`），与 swish 模块的
`swish / swish_half` 习惯一致。

### 1. `elu_f32`（FP32 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = elu(x[idx]);
}
```

- 一个 work-item 只算一个元素，正确性最直观，是后面所有版本的基准；
- `if (idx < N)` 处理 `N` 不能被 work-group 覆盖时“多启动”的 work-item；
- `sycl::exp` 是 float 版本（对应 CUDA 的 `expf`；写成 `sycl::exp` 传 double
  会提升成 double 计算，明显更慢）。

### 2. `elu_f32x4`（FP32 向量化版）

```cpp
int idx = 4 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  float4_vec reg_x = LOAD_FLOAT4(x + idx);   // 一次读 16 字节
  float4_vec reg_y;
  reg_y[0] = elu(reg_x[0]);
  reg_y[1] = elu(reg_x[1]);
  reg_y[2] = elu(reg_x[2]);
  reg_y[3] = elu(reg_x[3]);
  STORE_FLOAT4(y + idx, reg_y);              // 一次写 16 字节
}
```

- 一次 16 字节的向量 load/store 代替 4 次 4 字节访存，减少访存指令条数；
- 与 elementwise 不同，ELU 是**单输入**算子，没有 `b` 侧的第二路 load；
- “向量化”省的只是访存指令：计算仍要按 4 个分量逐条展开，每个分量都可能触发
  一次 `exp`，属于计算偏重的算子，所以收益不如 elementwise 加法明显；
- 读取写在 `if (idx < N)` **内部**（照搬 CUDA 版 `elu_f32x4_kernel`）。

### 3. `elu_f16`（FP16 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = elu_half(x[idx]);
}
```

- 数据类型为 `sycl::half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- half 没有隐式类型提升：比较用 `>`、乘法用 `*`、减法用 `-`、指数用 `sycl::exp`，
  常数用 `sycl::half(...)` 显式转换；
- 负半轴 `exp(x) - 1` 在 `x` 接近 0 时有**相减抵消**：`exp(x) ≈ 1`，
  两者相减后有效位数明显减少（FP32 同样存在，只是尾数多、影响小）。

### 4. `elu_f16x2`（FP16 每 work-item 2 个元素）

```cpp
int idx = 2 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  half2_vec reg_x = LOAD_HALF2(x + idx);
  half2_vec reg_y;
  reg_y[0] = elu_half(reg_x[0]);
  reg_y[1] = elu_half(reg_x[1]);
  STORE_HALF2(y + idx, reg_y);
}
```

- 一次用 `sycl::vec<sycl::half,2>`（2 个 half = 4 字节）完成两个元素的搬运；
- **与 relu 的 f16x2 不同**：ReLU 有成对指令 `__hmax2`，循环步长可以写成 2；
  ELU 没有 `hexp2` 这样的成对指数指令，只能逐分量调用 `elu_half`。

### 5. `elu_f16x8`（FP16 每 work-item 8 个元素，unpack 写法）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half2_vec reg_x_0 = LOAD_HALF2(x + idx + 0);
half2_vec reg_x_1 = LOAD_HALF2(x + idx + 2);
half2_vec reg_x_2 = LOAD_HALF2(x + idx + 4);
half2_vec reg_x_3 = LOAD_HALF2(x + idx + 6);
...
if ((idx + 0) < N) { STORE_HALF2(y + idx + 0, reg_y_0); }
if ((idx + 2) < N) { STORE_HALF2(y + idx + 2, reg_y_1); }
if ((idx + 4) < N) { STORE_HALF2(y + idx + 4, reg_y_2); }
if ((idx + 6) < N) { STORE_HALF2(y + idx + 6, reg_y_3); }
```

- 每个 work-item 处理 8 个 half，拆成 4 个 `vec<half,2>`（`idx+0/2/4/6`）分别处理；
- 这里的 “unpack” 指：数据在内存里本就是连续的 8 个 half，
  本 kernel 不把它整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 4 次读取是**无条件执行**的（越界判断只出现在写回处），照搬 CUDA 版。

### 6. `elu_f16x8_pack`（128 位打包版）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half8_vec pack_x = LOAD_HALF8(x + idx);   // 一次读 128 位
half8_vec pack_y;
#pragma unroll
for (int i = 0; i < 8; ++i) {
  pack_y[i] = elu_half(pack_x[i]);        // 逐个 half，没有成对指令可用
}
if ((idx + 7) < N) {
  STORE_HALF8(y + idx, pack_y);           // 一次写 128 位
}
```

- 8 个 half 恰好是 16 字节 = 128 位，用 `sycl::vec<sycl::half,8>` 一次 load/store，
  比 Kernel 5 的多条 4 字节访存指令更少；
- 与 elementwise / sigmoid / relu 的 `f16x8_pack` 一样，这里的 “pack” 指的是把连续
  8 个 half 整体当作 128 位向量搬运；计算部分仍是按 half 粒度处理；
- **循环步长的横向对照**很有教学意义：

  | 模块 | 成对指令 | pack 版循环 |
  | --- | --- | --- |
  | `relu` | 有 `__hmax2`（SYCL：`sycl::fmax` 的 `vec<half,2>` 重载） | `i += 2`，一次 2 个 half |
  | `sigmoid` | 无 `hexp2` | `i++`，逐个 half |
  | `elu` | 无成对指数指令 | `i++`，逐个 half |
  | `swish` | 无成对指数/除法 | `i++`，逐个 half |
  | `gelu` | 无 `htanh2` | `i++`，逐个 half |

## 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部（tail）标量回退分支；本模块的向量化版本**两个都没有**：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `elu_f32x4` | `idx < N` | 最后一个 work-group 的部分 work-item 会把 `idx+1..idx+3` 写到 `y` 合法范围之外 |
| `elu_f16x2` | `idx < N` | 同上，越界写 1 个 half |
| `elu_f16x8` | `(idx + 0/2/4/6) < N` | 同上，越界写 1 个 half |
| `elu_f16x8_pack` | `(idx + 7) < N` | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

注意 `elu_f32x4` 的判断是 `idx < N`（CUDA 版如此），与 `relu_f32x4` 的
`(idx + 0) < N` 写法等价、与 `gelu_f32x4` 的“先无条件读再判断写”不同。
本模块基准测试的 `S` / `K` 都取 1024 的倍数，启动的 work-item 恰好整段对齐，
不会暴露该问题。

如需在生产中使用，应参照 elementwise 补充尾部处理：

```cpp
if ((idx + 3) < N) {
  // 4 个元素都有效，走向量化分支
} else if (idx < N) {
  // 尾部不足 4 个元素，退回逐元素处理
}
```

### FP16 的两处精度陷阱（公式本身的性质，不是 bug）

1. **`exp(x) - 1` 的相减抵消**：`x` 是接近 0 的负数时 `exp(x) ≈ 1`，
   相减后有效位数大量丢失。例如 `x = -0.001` 时 `exp(x) ≈ 0.999`，
   `0.999 - 1 = -0.001`，在 10 位尾数下相对误差被放大；
2. **负半轴饱和**：`x` 小到大约 -11 时 `sycl::exp(x)` 已经下溢成 0，
   结果直接取到 `-alpha`（FP32 要到 -88 左右才下溢）。
   基准测试用 `torch.randn` 生成输入（几乎都落在 ±5 内），不会触发这一饱和。

### 与 PyTorch 的一致性

- FP32：`elu_f32` / `elu_f32x4` 与 `torch_elu`（以及 `torch.nn.functional.elu`）
  在相同输入上应当逐位一致；
- FP16：二者都受上面两处精度陷阱影响，允许约 `1/1024` 量级的量化差异；
  本模块的 `torch_elu` 在 half 张量上做 `torch.exp` / `torch.where`，
  与 kernel 的 half 路径并不完全相同，末位差异属于预期；
- **NaN**：ELU 的分支条件让 NaN 正常传播，这一点与 PyTorch 一致，
  是本仓库里少数“NaN 语义没问题”的模块（relu / gelu 都会吞掉 NaN）。实测对照：

  | 输入 `x` | `elu_f32` | `relu_f32` |
  | --- | --- | --- |
  | NaN | NaN | 0.0 |
  | -inf | -1.0（即 `-alpha`） | 0.0 |
  | +inf | inf | inf |
  | -100 | -1.0 | 0.0 |

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

对应代码为 `local = K / n_elements`、`global = S * local`；行过长或不可整除时
回退到上面的展平策略。

> 与 CUDA 版的差异：CUDA 的 `TORCH_BINDING_ELU` 只判断了 `(K / n_elements) <= 1024`，
> 没有“大于 0”与“能被整除”两项检查。若 `K` 不能被向量宽度整除，CUDA 版会按
> “一行一个 block”启动，导致每行末尾几个元素永远不被处理（静默漏算）。
> SYCL 移植版本沿用 elementwise / relu / sigmoid 模块的 3 项检查，
> 这种情况回退到展平启动；本模块基准的形状都能整除，两种写法启动配置完全一致。

各版本的 work-group 大小（`local`）：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每个 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `elu_f32` | 1 | 256 | 256 |
| `elu_f32x4` | 4 | 64 | 256 |
| `elu_f16` | 1 | 256 | 256 |
| `elu_f16x2` | 2 | 128 | 256 |
| `elu_f16x8` | 8 | 32 | 256 |
| `elu_f16x8_pack` | 8 | 32 | 256 |

## PyTorch 绑定

`elu.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python，
与 CUDA 版一一对应。主要工作：

1. 用宏检查 `x` / `y` 的 dtype 与 XPU 设备，并检查两者形状一致；
2. 根据维度/形状计算 nd_range 的 global / local；
3. 用 `data_ptr()` 拿裸指针（XPU 张量内存即 SYCL USM 内存，可直接交给 kernel）；
4. 从当前 XPU stream 取 `sycl::queue`，`queue.submit` 提交 kernel lambda。

调用链（以 `elu_f32` 为例）：

```text
elu.py
  └─ lib.elu_f32(x, y)                        # Python 调用（pybind11）
      └─ elu_f32(x, y)                        # C++ host 包装函数（宏生成）
          └─ submit_elu_f32(queue, ...)       # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)   # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

本模块与其它逐元素模块的 host 函数签名完全一致（都是单输入算子）：

| 模块 | host 函数签名 | kernel 参数 |
| --- | --- | --- |
| elementwise | `elementwise_add_xxx(a, b, c)` | `a, b, c, N` |
| sigmoid | `sigmoid_xxx(x, y)` | `x, y, N` |
| relu | `relu_xxx(x, y)` | `x, y, N` |
| elu | `elu_xxx(x, y)` | `x, y, N` |

## 基准测试脚本

`elu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `elu.sycl`；
2. 打印扩展构建目录与 XPU 设备名（与 elementwise / histogram / relu / sigmoid 对齐）；
3. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合在 XPU 上生成随机张量
   （FP32 与 FP16 各一套，显式 `.contiguous()`）；
4. `run_benchmark` 先 warmup，再运行 1000 次取平均；
5. 依次对比 6 个自定义 kernel 与 `torch_elu` 的正确性和耗时。

官方对照的写法与 CUDA 版一致，直接用 `torch_elu`（**保留 `out` 参数**，
结果原地写进 `y`）：

```python
def torch_elu(x, out=None):
    if out is None:
        return torch.where(x > 0, x, 1.0 * (torch.exp(x) - 1))
    else:
        torch.where(x > 0, x, 1.0 * (torch.exp(x) - 1), out=out)
        return out
```

这个对照由 `gt / exp / sub / mul / where` 多个 elementwise kernel 组成，
反复读写显存，所以耗时明显高于单个融合 kernel；真正公平的对照应该是
`torch.nn.functional.elu`——本次保持 CUDA 版脚本原样，只在文档里说明。

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（`torch.xpu`，实测版本 2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

与其它模块相同，`elu.py` 在 `load(...)` 前执行
`os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`：

- 未显式设置时只生成 `-fsycl-targets=spir64`（JIT），由 Intel GPU 驱动即时编译；
- 不需要 `ocloc`，任何 Intel GPU 都能跑，缺点是首次运行有编译/加载开销；
- 若机器已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 与 CUDA 版的对照：CUDA 版脚本用 `--use_fast_math`（ELU 的负半轴 `exp` 收益明显，
> 它同时会打开 FTZ，极小的结果会被直接清成 0），SYCL 侧默认用
> 精度完整的 `sycl::exp`；如需对齐可改用 `sycl::native::exp`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 elu.py`，结果为：

- FP32 三个版本（`elu_f32` / `elu_f32x4` / `torch_elu`）在同一份输入上一致；
- FP16 各版本彼此一致；FP16 与 FP32 之间存在约 `1/1024` 量级的量化差异，
  负半轴靠近 0 的位置还会叠加 `exp(x) - 1` 的相减抵消；
- 结果都落在 `(-1, +∞)` 内（alpha = 1），没有出现 `NaN` / `inf`；
- 输出的正负分布取决于随机输入：`torch.randn` 约有一半元素为负，
  故前 2 个元素经常是小负数（如 `[-0.44692516, 0.08315884]`）。

完整输出见 `kernels/elu/README.md`。

### Arc A770 实测（S=4096, K=4096，JIT/spir64，一次运行结果）

ELU 是单输入算子：每个元素读 1 份、写 1 份，所以等效访存量 = `2 × N × sizeof(dtype)`；
本组 `N = 16,777,216`，FP32 为 134.2 MB，FP16 为 67.1 MB。

| 版本 | 耗时 (ms) | 等效带宽 |
| --- | ---: | ---: |
| f32 | 0.3225 | ~416 GB/s |
| f32x4 | 0.3276 | ~410 GB/s |
| torch f32（对照） | 1.7232 | ~78 GB/s |
| f16 | 0.2897 | ~232 GB/s |
| f16x2 | 0.1710 | ~393 GB/s |
| f16x8 | 0.3298 | ~204 GB/s |
| f16x8_pack | 0.1751 | ~383 GB/s |
| torch f16（对照） | 0.9484 | ~71 GB/s |

由此得到的结论：

1. **FP32 三个版本都停在 ~410~416 GB/s**，与 relu / sigmoid 在同一台机器上
   测到的上限完全一致；
2. **本模块最值得看的一条对照**：`elu_f32` 0.3225 ms 与 `relu_f32` 0.3221 ms、
   `sigmoid_f32` 0.3222 ms 几乎相同——ELU 每个元素多了一次负半轴的 `exp`，
   但在 16.7M 元素这个规模下它完全被访存掩盖了；
3. FP16 最快版本比 FP32 标量版快约 **1.9 倍**（0.3225 ms → 0.1710 ms），
   与“访存字节数减半”的预期一致；
4. FP16 内部差异明显：`f16x2`（~393 GB/s）与 `f16x8_pack`（~383 GB/s）最快，
   而 `f16x8`（unpack，把 8 个元素拆成 4 次 4 字节访存）只有 ~204 GB/s，
   甚至比 `f16` 标量版（~232 GB/s）还慢——这与 elementwise / relu / sigmoid
   在 Arc A770 上的结论一致；
5. 向量化对 FP32 没有收益（`f32x4` 0.3276 ms ≈ `f32` 0.3225 ms），
   符合“已经吃满带宽时，减少访存指令条数不再带来额外收益”的判断；
6. 对照实现 `torch_elu` 只有 ~78 GB/s（FP32）/ ~71 GB/s（FP16），因为它由
   `gt / exp / sub / mul / where` 多个 kernel 组成、反复读写显存，
   不是公平对比（应用 `torch.nn.functional.elu`）；
7. 小规模（如 `S=1024, K=1024`）时各版本差异变小、甚至出现波动，
   因为 kernel 时间已经接近启动开销量级。

## 后续可以尝试的优化方向

- 补上 `f32x4` / `f16x2` / `f16x8` 的 tail 分支与 `f16x8_pack` 的尾部回退，
  使任意 `N` 都安全（对齐 LeetOneAPI elementwise 模块的写法）；
- 对比 `sycl::exp` 与 `sycl::native::exp`（快速近似，对应 CUDA 的
  `--use_fast_math`）的精度与速度差异——ELU 的开销主要压在负半轴的 `exp` 上；
- 换用 `torch.nn.functional.elu` 做公平对照，替换掉脚本自拼的 `torch_elu`；
- 试试用 `expm1(x)` 改写负半轴（`exp(x) - 1` 的相减抵消会明显改善），
  对比精度与性能；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 对比不同 work-group 大小、不同向量宽度对带宽/吞吐的实测影响。
