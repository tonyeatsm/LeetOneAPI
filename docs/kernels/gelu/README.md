# GELU（Gaussian Error Linear Unit，高斯误差线性单元）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 gelu（高斯误差线性单元）教学示例移植到 Intel oneAPI。
PyTorch 支持两种形式（`torch.nn.GELU` 的 `approximate` 参数）：

```text
精确式（approximate='none'）：
  gelu(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
tanh 近似（approximate='tanh'）：
  gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

y[i] = gelu(x[i]),  i = 0 .. N-1
```

其中 `Φ(x)` 是标准正态分布的累积分布函数（CDF）。本模块与 CUDA 版一样默认走
**tanh 近似**（`GELU_OPS` / `HALF_GELU_OPS` 宏指向 `gelu_tanh_approximate`），
同时保留 FP32 的精确式实现 `gelu_none_approximate`（用 `erf`），改一行宏定义就能切换。

在 CUDA 版中，同一个运算有 6 种写法；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `gelu_f32`：FP32 标量版（每 work-item 1 个元素）
2. `gelu_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `gelu_f16`：FP16 标量版（`sycl::half`）
4. `gelu_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `gelu_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存，unpack 写法）
6. `gelu_f16x8_pack`：FP16 每 work-item 8 个元素（128 位打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> 计算逻辑位于源码 `submit_gelu_*` 函数里 `queue.submit + parallel_for` 提交的
> kernel lambda 中；下文为便于教学对照，仍按 6 个“kernel 版本”讲述。

**与前面几个模块的关系**：6 个版本与 elementwise / sigmoid / relu / elu 一一对应，
区别只是算子。GELU 在 relu / elu / sigmoid / gelu / swish 这一族里**计算量最大**：
一次 tanh（内部还含一次 exp）外面还套着 `x` 的三次多项式。

**本模块的独特价值**：它是全仓库里**精度陷阱最深**的模块，三个问题都值得单独看：

1. half 没有 `tanh`，只能用 `(exp(2t) - 1) / (exp(2t) + 1)` 拼，
   于是 `exp` 的溢出会变成 `inf / inf = NaN`——所以必须 clamp；
2. 但 CUDA 版 clamp 的是**输入 `x`**，真正送进 `exp` 的是 `2 * inner`（含 `x³`），
   于是 FP16 版本只要 `x ≳ 4.03` 就输出 **NaN**（这是已识别、本次不修复的缺陷）；
3. `inner` 接近 0 时分子是“两个相近的数相减”，半精度下有效位数大量丢失，
   即便不溢出，FP16 结果也比 PyTorch 明显粗糙。

与同族算子的对照：

| 对比项 | relu | elu（alpha=1） | gelu（tanh 近似） | swish / silu |
| --- | --- | --- | --- | --- |
| 公式 | `max(0, x)` | `x>0 ? x : exp(x)-1` | `0.5x(1+tanh(k(...)))` | `x/(1+exp(-x))` |
| 值域 | `[0, +∞)` | `(-1, +∞)` | `(-0.17, +∞)` | `(-0.278, +∞)` |
| 单调性 | 单调 | 单调 | 非单调（负半轴凹陷） | 非单调 |
| `x = 0` 处 | 不可导 | 可导，导数 1 | 光滑可导 | 光滑可导，导数 0.5 |
| 每元素计算 | 一次取最大值 | 一次比较 + 一次 `exp` | 多次乘加 + 一次 `tanh` | `exp` + 加/除/乘 |
| 溢出保护 | 不需要 | 不需要 | **需要 clamp** | 不需要 |
| 成对指令 | 有 `__hmax2` | 无 | 无 `htanh2` | 无 |

最终目标不是追求极致性能，而是让学习者直观理解“**exp 出现在什么位置决定了要不要保护**”
这条经验：在分母上安全（swish）、只在负半轴出现也安全（elu），
而自己除自己（gelu 的 tanh）最危险。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/gelu/gelu.cu` 的行为，**只做 CUDA → SYCL
的等价移植，不改算法逻辑**，具体包括：

1. 复刻 6 个 kernel 的写法与 host 端启动配置；
2. 复刻 `MAX_EXP_F32` / `MIN_EXP_F32` / `MAX_EXP_F16` / `MIN_EXP_F16` 四个 clamp 边界
   与 `v = clamp(x)` 的写法（**照搬 CUDA 版夹的是输入 `x`，不是 `inner`**）；
3. 复刻系数宏：`SQRT_2_PI`（≈ 0.7978845608）、`HALF_SQRT_2_PI`、`HALF_V_APP`（0.044715）
   与 `HALF_1` / `HALF_2` / `HALF_DIV2`，以及 `GELU_OPS` / `HALF_GELU_OPS` 算法开关；
4. 保留 FP32 的精确式 `gelu_none_approximate`（`erf` 版）作为可切换选项，
   但默认仍走 tanh 近似（与 CUDA 版一致）；
5. **原样保留**上面说的 FP16 溢出 NaN、FP32 截断 / 吞 NaN，以及向量化版本
   缺少 tail 分支等已知缺陷，仅在本文档与源码注释中照实标注；
6. 测试脚本的官方对照**照搬 CUDA 版**：`partial(torch.nn.GELU("tanh"))`。
   必须显式写 `"tanh"` 才能和本模块的 tanh 近似对齐（默认是 `approximate='none'`，
   即 erf 精确式）。该模块实例没有 `out` 参数，所以对照测试走
   `run_benchmark` 的“不传 out”分支；
7. 打印格式沿用 LeetOneAPI 现有模块（CUDA 版 `gelu.py` 本来也是干净格式，
   没有 `f"{v:<12}"` 的补齐）；
8. 按已有模块的组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/gelu/gelu.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `gelu.cu`，代码内含详细注释） |
| `kernels/gelu/gelu.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/gelu/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/kernels/gelu/README.md` | 本文件，模块设计说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid` / `block` | `nd_range<1>` 的 global / local range | 总 work-item 数 = grid × block |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<sycl::half,2>` | SIMD 向量类型 |
| `half` | `sycl::half` | FP16 类型，2 字节 |
| `tanhf` / `hexp` 拼 tanh | `sycl::tanh`（`float`）/ `sycl::exp` 拼 tanh（`sycl::half`） | SYCL 的 `sycl::half` **同样没有** `tanh`，必须照搬 exp 拼法 |
| `erff`（精确式） | `sycl::erf` | 仅 FP32 精确式用 |
| `fminf` / `fmaxf` / `__hmin` / `__hmax` | `sycl::fmin` / `sycl::fmax`（float 与 `sycl::half` 都有重载） | clamp |
| `__float2half(1.0f)` | `sycl::half(1.0f)` | 常数转 half |
| `LDST128BITS(value)`（`reinterpret_cast<float4*>`） | `sycl::vec<sycl::half,8>` 的 load/store 宏 | 128 位整体访存 |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构；本示例默认置空走 JIT(`spir64`) |

## GELU 曲线与数学性质

```text
  y = x * Φ(x)（tanh 近似画法，x ∈ [-3, 3]）

4.0 ┤                          ***
    ┤                       ***
3.0 ┤                    ***
    ┤                 ***
2.0 ┤              ***
    ┤           ***
1.0 ┤        ***
0.0 ┤*******+                     ← 唯一的零点在 x = 0
    ┤    ***
-0.17┤   *                        ← 负向小凹陷，min ≈ -0.1700（x ≈ -0.752）
    └──────────────────────────────→ x
    -3   -2   -1   0   1   2   3

采样值(tanh 近似)：-3→-0.0036、-2→-0.0454、-1→-0.1588、-0.75→-0.1700、
                  0→0、1→0.8412、2→1.9546、3→2.9964
```

读图要点：

- **正半轴近似恒等映射**：`x` 稍大（约 3 以上）后 `Φ(x) ≈ 1`，`gelu(x) ≈ x`；
- **负半轴不是硬零**：`x < 0` 时输出是小负数，最小值约 `-0.1700`
  （出现在 `x ≈ -0.752`），所以 GELU **不是单调函数**；
- **在 `x = 0` 处光滑可导**（曲线整体无穷阶可导）：既没有 ReLU 的“0 点不可导”，
  也没有 ELU 的“负半轴带一个转折”；
- **负半轴梯度不为 0**：和 ELU 一样避免了死亡 ReLU，但负值区间更浅
  （最深 -0.17，ELU 可以一路滑到 -1）；
- **计算量最大**：一次 `tanh`（内部还要一次 `exp`）加若干次乘加，
  在本仓库这一族算子里属于计算偏重的算子。

## 数值稳定性：为什么 GELU 必须 clamp，以及 clamp 该夹谁

half 没有 `tanh`，CUDA 版（SYCL 版同理）用恒等式拼出来：

```text
tanh(t) = (exp(2t) - 1) / (exp(2t) + 1)
```

只要 `2t` 超过 `exp` 的溢出阈值，`exp(2t)` 就变成 `inf`，分子分母同时变成 `inf`，
`inf / inf = NaN`。所以必须先把输入夹进安全区间——这与 sigmoid 的思路一致：

| 精度 | 常数 | 数值 | 含义 |
| --- | --- | --- | --- |
| FP32 | `MAX_EXP_F32` / `MIN_EXP_F32` | `±88.3762626647949` | 逐元素 sigmoid 常用的保守上界，略小于 `ln(FLT_MAX) ≈ 88.723` |
| FP16 | `MAX_EXP_F16` | `11.089866488461016` | `ln(65504)`，65504 是 half 的最大规格数 |
| FP16 | `MIN_EXP_F16` | `-9.704060527839234` | `ln(2^-14)`，`2^-14` 是 half 的最小规格正数 |

**这里有一个必须讲清楚的坑**：CUDA 版夹的是**输入 `x`**，而真正送进 `exp` 的是
`2 * inner`，其中

```text
inner = sqrt(2/pi) * (x + 0.044715 * x^3) ≈ 0.797885 * (x + 0.044715 * x^3)
```

反解 `2 * inner = ln(65504)` 得到 `x ≈ 4.0278`：也就是说 FP16 版本只要输入略大于
**4.03**，`exp` 就会溢出并算出 **NaN**（CUDA 版实测 `x = 4.0` 正常、`x = 4.1` 为 NaN）。
正确的修法是把 clamp 的边界换算到 `inner` 上，或者干脆把 half 提升成 float 再算 tanh——
本模块照搬 CUDA 版，**不修复**，只在文档与注释里标注。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。

### 0. 系数宏与三个辅助函数

```cpp
#define SQRT_2_PI (M_SQRT2 * M_2_SQRTPI * 0.5f)          // ≈ 0.7978845608 = sqrt(2/pi)
#define HALF_SQRT_2_PI (sycl::half(M_SQRT2) * sycl::half(M_2_SQRTPI) * HALF_DIV2)
#define HALF_V_APP sycl::half(0.044715f)
#define HALF_GELU_OPS gelu_tanh_approximate              // FP16 走哪个函数
#define GELU_OPS gelu_tanh_approximate                   // FP32 走哪个函数
```

三个 `static inline` 辅助函数（对应 CUDA 的 `__inline__ __device__`）：

```cpp
// FP16：用 exp 拼 tanh——两个精度陷阱都在这一行里
static inline sycl::half gelu_tanh_approximate(sycl::half x) {
  sycl::half x_cube = x * x * x;
  sycl::half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  return HALF_DIV2 * x *
         (HALF_1 + ((sycl::exp(inner * HALF_2) - HALF_1) /
                    (sycl::exp(inner * HALF_2) + HALF_1)));
}

// FP32：直接用 tanh，不存在上面两个问题
static inline float gelu_tanh_approximate(float x) {
  return 0.5f * x * (1.0f + sycl::tanh(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// FP32 精确式（approximate='none'），把 GELU_OPS 改成它即可切换
static inline float gelu_none_approximate(float x) {
  return x * 0.5f * (1.0f + sycl::erf(x * M_SQRT1_2));
}
```

> 与 CUDA 版的差异：CUDA 的 `SQRT_2_PI` 宏没加括号（靠“乘法同优先级、从左到右”
> 恰好算对），SYCL 版补上了括号。另外 CUDA 版文件里紧邻的英文注释把系数写成
> `sqrt(2*pi)/2`（≈1.2533，其实不对，正确等价写法是 `2/sqrt(2*pi)`），
> 但代码取值是对的；SYCL 版按正确写法注释。
>
> SYCL 侧还有一个必须处理的差异：`SQRT_2_PI` 里的 `M_SQRT2` / `M_2_SQRTPI` 是
> **double** 类型常量，若照抄 CUDA 的写法，整个表达式就是 double，`sycl::tanh`
> 会解析到 double 重载，在 Intel GPU 上直接报
> `Required aspect fp64 is not supported on the device`（Arc A770 没有 fp64）。
> 本模块把两个常量显式 `static_cast<float>` 后相乘，保证全程 float——这是
> CUDA 版不会遇到、SYCL 版必须踩一次的坑。

### 1. `gelu_f32`（FP32 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  // 先 clamp 再算
  float v = sycl::fmin(sycl::fmax(x[idx], MIN_EXP_F32), MAX_EXP_F32);
  y[idx] = GELU_OPS(v);
}
```

- **先 clamp 再算**：保证后续 `tanh` 的实参不会落到无穷 / NaN 的危险区；
- clamp 用的是 `sycl::fmin` / `sycl::fmax`（`float` 重载）；
- 已知副作用：`fmax(NaN, MIN_EXP_F32) = MIN_EXP_F32`（IEEE `maxNum` 语义），
  所以 NaN 输入会被“洗”成 -88.376 并最终输出 0，**与 PyTorch 不一致**；
  同时 `|x| > 88.376` 的输入会被截断（`gelu_f32(100) = 88.376`）。

### 2. `gelu_f32x4`（FP32 向量化版）

```cpp
int idx = 4 * static_cast<int>(item.get_global_id(0));
float4_vec reg_x = LOAD_FLOAT4(x + idx);   // 无条件读入 16 字节
float4_vec reg_y;
#pragma unroll
for (int i = 0; i < 4; ++i) {
  reg_x[i] = sycl::fmin(sycl::fmax(reg_x[i], MIN_EXP_F32), MAX_EXP_F32);
}
#pragma unroll
for (int i = 0; i < 4; ++i) {
  reg_y[i] = GELU_OPS(reg_x[i]);
}
if ((idx + 0) < N) {
  STORE_FLOAT4(y + idx, reg_y);            // 只在写回时判断越界
}
```

- **写法与其它模块不同**：这里没有在 load 之前做越界判断，
  而是先无条件读入 16 字节，最后只在写回时判断 `(idx + 0) < N`；
- GELU 的算术强度高（一次 `tanh` 内部就含一次 `exp`），访存指令数不是瓶颈，
  所以向量化那点收益容易被“逐分量展开的冗长计算”吃掉；
- 读取在判断之外，因此 `N` 不是 4 的倍数时还会**越界读**（见边界约束）。

### 3. `gelu_f16`（FP16 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  sycl::half v = x[idx];
  v = sycl::fmin(sycl::fmax(v, MIN_EXP_F16), MAX_EXP_F16);   // half 版 clamp
  y[idx] = HALF_GELU_OPS(v);
}
```

- half 没有隐式类型提升，常数要用 `sycl::half(...)` 显式转换；
- clamp 用 `sycl::fmin` / `sycl::fmax` 的 `sycl::half` 重载（对应 CUDA 的
  `__hmin` / `__hmax`），上下界不对称（见上表）；
- 计算走 `HALF_GELU_OPS`，也就是用 `exp` 拼 `tanh` 的近似实现——
  它既有“相减抵消”的精度损失，也有“`exp` 溢出 → `inf/inf = NaN`”的正确性问题。

### 4. `gelu_f16x2`（FP16 每 work-item 2 个元素）

```cpp
int idx = 2 * static_cast<int>(item.get_global_id(0));
half2_vec reg_x = LOAD_HALF2(x + idx);     // 无条件读入 4 字节
half2_vec reg_y;
reg_x[0] = sycl::fmin(sycl::fmax(reg_x[0], MIN_EXP_F16), MAX_EXP_F16);
reg_x[1] = sycl::fmin(sycl::fmax(reg_x[1], MIN_EXP_F16), MAX_EXP_F16);
reg_y[0] = HALF_GELU_OPS(reg_x[0]);
reg_y[1] = HALF_GELU_OPS(reg_x[1]);
if ((idx + 0) < N) {
  STORE_HALF2(y + idx, reg_y);
}
```

- 同样先无条件读入、只在写回时判断越界；
- **与 relu 的 f16x2 不同**：ReLU 有成对指令 `__hmax2`；GELU 没有 `htanh2`，
  只能逐分量计算。

### 5. `gelu_f16x8`（FP16 每 work-item 8 个元素，unpack 写法）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half2_vec reg_x_0 = LOAD_HALF2(x + idx + 0);
half2_vec reg_x_1 = LOAD_HALF2(x + idx + 2);
half2_vec reg_x_2 = LOAD_HALF2(x + idx + 4);
half2_vec reg_x_3 = LOAD_HALF2(x + idx + 6);
// 8 个分量逐个 clamp，再逐个算 GELU
...
```

- 每个 work-item 处理 8 个 half，拆成 4 个 `vec<half,2>`；
- **排版上的小细节**：计算时把结果又写回了 `reg_x_*`（复用了输入寄存器），
  而不是像其它模块那样用单独的 `reg_y_*` 接收结果——本次只补充注释，不改逻辑
  （SYCL 版仍声明了 `reg_y_*` 以保持与 CUDA 版逐行对应）；
- 4 次读取是无条件执行的，越界判断只出现在写回处。

### 6. `gelu_f16x8_pack`（128 位打包版）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half8_vec pack_x = LOAD_HALF8(x + idx);   // 一次读 128 位
half8_vec pack_y;
#pragma unroll
for (int i = 0; i < 8; ++i) {
  sycl::half v = sycl::fmin(sycl::fmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
  pack_y[i] = HALF_GELU_OPS(v);
}
if ((idx + 7) < N) {
  STORE_HALF8(y + idx, pack_y);           // 一次写 128 位
}
```

- 8 个 half 恰好 16 字节 = 128 位，一次 load/store 完成搬运，比 unpack 版指令更少，
  实测也是 FP16 里最快的（CUDA 侧结论）；
- 循环写成 `++i` 逐个 half 算——GELU 没有成对指令（对照 `relu` 的 `i += 2`）。

## 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

### 1. FP16 版本在 `x ≳ 4.03` 时输出 NaN（最严重）

原因见上文“clamp 该夹谁”。这是**正确性**问题而不是精度问题：

```text
gelu_f16(3.9)  = 3.900390625
gelu_f16(4.0)  = 3.998046875   （还没溢出）
gelu_f16(4.03) = NaN           （exp(2*inner) 已经溢出成 inf → inf/inf）
gelu_f16(4.1)  = NaN
```

实测阈值落在 4.0 ~ 4.03 之间，与理论值 x ≈ 4.0278（反解 2*inner = ln(65504)）吻合。

基准测试用 `torch.randn` 生成输入，大尺寸下**会出现个别** `|x| > 4.03` 的元素，
所以 FP16 那几组的输出里可能混着 NaN——而打印只看前 2 个元素，抽查时看不出来。
要确认可以把 `x` 整体 `torch.abs().max()` 打出来对照。

### 2. FP32 版本对大输入会截断，且会吞掉 NaN

`v = fmin(fmax(x, -88.376), 88.376)` 的副作用：

| 输入 | kernel 输出 | PyTorch 输出 |
| --- | --- | --- |
| `x = 100` | 88.376 | 100（近似恒等） |
| `x = -100` | -0.0036 附近（被夹到 -88.376 后算 GELU） | ≈ 0 |
| `x = NaN` | -0.0 | NaN |

本模块基准输入的 `|x|` 远小于 88.376，不会触发。

### 3. FP16 的 tanh 实现精度低于 PyTorch

即便不溢出，`(exp(2t)-1)/(exp(2t)+1)` 在 `t ≈ 0` 附近也存在相减抵消，
所以 FP16 结果与 `torch.nn.GELU("tanh")` 之间会有比其它模块更明显的末位差异。
PyTorch 内部会把 half 提升到 float 再算 tanh，本模块则在 half 上直接算。

### 4. 向量化版本的尾部（tail）分支缺失

elementwise 用 `(idx + 3) < N` 判断整段是否齐全并配了尾部回退；本模块没有：

| kernel | 越界判断 | `N` 不是向量宽度整数倍时 |
| --- | --- | --- |
| `gelu_f32x4` | 读：无；写：`(idx + 0) < N` | 越界读 1~3 个 float，`idx < N` 时还会越界写 1~3 个 float |
| `gelu_f16x2` | 读：无；写：`(idx + 0) < N` | 同上（1 个 half） |
| `gelu_f16x8` | 读：无；写：`(idx + 0/2/4/6) < N` | 同上（1 个 half） |
| `gelu_f16x8_pack` | 读：无；写：`(idx + 7) < N` | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S` / `K` 都取 1024 的倍数，启动的 work-item 恰好整段对齐，
不会暴露该问题。

## 启动配置（host 端）

host 端与 CUDA 版保持同一套策略：

```cpp
// 非二维张量：展平启动
local_range  = ELEMS_PER_GROUP / n_elements;               // 256 / n_elements
global_range = ceil(N / 256) * local_range;

// 二维 (S, K)：每行放得下且能被整除时，一行一个 work-group
local_range  = K / n_elements;
global_range = S * local_range;
```

> 与 CUDA 版的差异：CUDA 的 `TORCH_BINDING_GELU` 只判断了 `(K / n_elements) <= 1024`，
> 没有“大于 0”与“能被整除”两项检查；SYCL 版沿用前面几个模块的 3 项检查。

各版本的 work-group 大小（`local`）：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每个 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `gelu_f32` | 1 | 256 | 256 |
| `gelu_f32x4` | 4 | 64 | 256 |
| `gelu_f16` | 1 | 256 | 256 |
| `gelu_f16x2` | 2 | 128 | 256 |
| `gelu_f16x8` | 8 | 32 | 256 |
| `gelu_f16x8_pack` | 8 | 32 | 256 |

## PyTorch 绑定

`gelu.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python：

1. 用宏检查 `x` / `y` 的 dtype 与 XPU 设备，并检查两者形状一致；
2. 根据维度/形状计算 nd_range 的 global / local；
3. 用 `data_ptr()` 拿裸指针；从当前 XPU stream 取 `sycl::queue` 提交 kernel lambda。

调用链（以 `gelu_f32` 为例）：

```text
gelu.py
  └─ lib.gelu_f32(x, y)                       # Python 调用（pybind11）
      └─ gelu_f32(x, y)                       # C++ host 包装函数（宏生成）
          └─ submit_gelu_f32(queue, ...)      # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)   # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

## 基准测试脚本

`gelu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `gelu.sycl`；
2. 打印扩展构建目录与 XPU 设备名；
3. 对 `S` / `K` 的 9 种组合生成随机张量（FP32 与 FP16 各一套）；
4. `run_benchmark` 先 warmup，再运行 1000 次取平均；
5. 依次对比 6 个自定义 kernel 与 PyTorch 官方的正确性和耗时。

官方对照与 CUDA 版一致，用 `partial(torch.nn.GELU("tanh"))`：

```python
run_benchmark(lib.gelu_f32, x, "f32", y)
run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
run_benchmark(partial(torch.gelu), x, "f32_th")      # 不传 out，走另一条分支
```

两点说明：

- `torch.nn.GELU("tanh")` 是 `nn.Module` **实例**，没有 `out` 参数，
  所以对照测试走 `run_benchmark` 的“不传 out”分支（脚本里把实例赋给了
  `torch.gelu` 这个名字，与 CUDA 版写法一致）；
- 必须显式写 `"tanh"`：PyTorch 默认是 `approximate='none'`（erf 精确式），
  与 kernel 的 tanh 近似会有可见差异。

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

`gelu.py` 在 `load(...)` 前执行 `os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`，
默认只生成 `-fsycl-targets=spir64`（JIT），不需要 `ocloc`；
若已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 与 CUDA 版的对照：CUDA 版脚本用了 `--use_fast_math`（GELU 的开销主要在 tanh 上，
> 收益明显；但它也让 half 版的溢出行为更“干脆”，`x ≳ 4.03` 时直接得到 NaN）；
> SYCL 侧默认用精度完整的 `sycl::tanh` / `sycl::exp`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 gelu.py`，结果为：

- FP32 三个版本（`gelu_f32` / `gelu_f32x4` / `torch.nn.GELU("tanh")`）在同一份输入上
  基本一致，差异只在 `--use_fast_math`（CUDA）与默认精度（SYCL）的实现路径上；
- FP16 各版本彼此一致，与 PyTorch 的差异比其它模块更大
  （半精度下 exp 拼 tanh 的相减抵消）；
- 需要特别留意：大尺寸的 FP16 组里会有个别 `|x| > 4.03` 的元素产生 NaN，
  抽查前 2 个元素看不出来（详见“当前边界约束”第 1 条）。

完整输出见 `kernels/gelu/README.md`。

### Arc A770 实测（S=4096, K=4096，JIT/spir64，一次运行结果）

GELU 是单输入算子：每个元素读 1 份、写 1 份，等效访存量 = `2 × N × sizeof(dtype)`；
本组 `N = 16,777,216`，FP32 为 134.2 MB，FP16 为 67.1 MB。

| 版本 | 耗时 (ms) | 等效带宽 |
| --- | ---: | ---: |
| f32 | 0.3313 | ~405 GB/s |
| f32x4 | 0.3409 | ~394 GB/s |
| torch f32（对照） | 0.3416 | ~393 GB/s |
| f16 | 0.2916 | ~230 GB/s |
| f16x2 | 0.1719 | ~390 GB/s |
| f16x8 | 0.3350 | ~200 GB/s |
| f16x8_pack | 0.1765 | ~380 GB/s |
| torch f16（对照） | 0.2856 | ~235 GB/s |

由此得到的结论：

1. **GELU 是本仓库里唯一能在大尺寸下看到“计算量”差异的模块**：
   FP32 的 0.3313 ms 比 relu / elu / sigmoid 的 ~0.322 ms 慢了约 3%，
   因为它的算术强度最高（一次 tanh 内部还含一次 exp，外面还有三次多项式）；
2. FP32 内部：`f32x4`（0.3409）与 `torch GELU("tanh")`（0.3416）基本持平，
   **都略慢于标量版**（0.3313）——向量化省下的访存指令被冗长的
   逐分量计算吃掉了，这是本仓库里少见的“向量化反而变慢”现象；
3. FP16 最快版本也是 `f16x2`（0.1719 ms，~390 GB/s），比 FP32 标量快约 1.9 倍；
4. `f16x8`（unpack）依然是 FP16 里最慢的（0.3350 ms，~200 GB/s），
   与其它模块的结论一致；
5. 对照实现 `torch.nn.GELU("tanh")` 走的是融合路径，所以它在 FP32 下与自定义
   kernel 基本持平（不像 `torch_elu` / `torch_swish` 那样明显偏慢）；
6. 注意这组数据里 FP16 部分混有 NaN（大尺寸下 `|x| > 4.03` 的元素），
   但它们不影响耗时统计的可信度；要验证请单独跑边界用例。

## 后续可以尝试的优化方向

- **优先修 NaN**：把 clamp 的边界换算到 `inner` 上（`2*inner ≤ ln(65504)`），
  或者在 FP16 路径上把 half 提升成 float 再算 tanh；
- 补上向量化版本的 tail 分支，使任意 `N` 都安全；
- 用 `sycl::native::exp` / 快速 tanh 对比精度与速度（对应 CUDA 的 `--use_fast_math`）；
- 切换 `GELU_OPS` 到 `gelu_none_approximate`，对比 erf 精确式与 tanh 近似的
  精度 / 性能差异，并与 `torch.nn.GELU("none")` 对照；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 对比不同 work-group 大小、不同向量宽度对带宽/吞吐的实测影响。
