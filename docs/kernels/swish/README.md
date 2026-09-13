# Swish（Swish / SiLU，自门控激活函数）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 swish（自门控激活函数）教学示例移植到 Intel oneAPI：

```text
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

y[i] = swish(x[i]),  i = 0 .. N-1
```

**命名说明**：swish 与 silu 是同一个函数的两个名字。Google 在 2017 年的论文里
把它叫 Swish（`x * sigmoid(beta * x)`，`beta = 1` 时即本模块的形式），
PyTorch 沿用了 SiLU 的叫法（`torch.nn.SiLU` / `torch.nn.functional.silu`）。
本模块沿用仓库里已有的 `swish` 命名。

在 CUDA 版中，同一个运算有 6 种写法；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `swish_f32`：FP32 标量版（每 work-item 1 个元素）
2. `swish_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `swish_f16`：FP16 标量版（`sycl::half`）
4. `swish_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `swish_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存，unpack 写法）
6. `swish_f16x8_pack`：FP16 每 work-item 8 个元素（128 位打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> 计算逻辑位于源码 `submit_swish_*` 函数里 `queue.submit + parallel_for` 提交的
> kernel lambda 中；下文为便于教学对照，仍按 6 个“kernel 版本”讲述。

**与前面几个模块的关系**：6 个版本与 elementwise / sigmoid / relu / elu / gelu 一一对应，
区别只是算子。Swish 可以看作 sigmoid 模块的“进阶版”：同样是 `exp`，
再多一步加法、除法和乘法。kernel 里没有真的先算 `sigmoid` 再乘，而是用等价的
`x / (1 + exp(-x))` 一次算完，省掉一次中间变量。

**本模块的独特价值**：它是“**溢出未必是坏事**”这条经验的最佳反例——
`exp(-x)` 溢出成 `inf` 时结果恰好是数学极限 `x * 0`，不需要任何 clamp；
而 gelu 把 `exp` 放进 `(exp(2t)-1)/(exp(2t)+1)` 这种自己除自己的形式就会得到 NaN。
同样是 `exp`，位置不同，风险完全不同。

与同族算子的对照：

| 对比项 | relu | gelu（tanh 近似） | swish / silu |
| --- | --- | --- | --- |
| 公式 | `max(0, x)` | `0.5x(1+tanh(...))` | `x / (1 + exp(-x))` |
| 值域 | `[0, +∞)`，仅负端截断 | `(-0.17, +∞)` | `(-0.2785, +∞)` |
| 最小值位置 | `x <= 0` 全为 0 | `x ≈ -0.752` | `x ≈ -1.279` |
| `x = 0` 处 | 不可导（左 0 右 1） | 光滑可导 | 光滑可导，导数 0.5 |
| 每元素计算 | 一次取最大值 | 多次乘加 + 一次 `tanh` | 一次 `exp` + 加/除/乘 |
| 溢出保护 | 不需要 | **需要 clamp** | 不需要（溢出对应极限值） |
| 成对指令 | 有 `__hmax2` | 无 `htanh2` | 无成对指数 / 除法 |

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及“同一个数学函数用不同写法实现，数值风险可以完全不同”。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/swish/swish.cu` 的行为，**只做 CUDA → SYCL
的等价移植，不改算法逻辑**，具体包括：

1. 复刻 6 个 kernel 的写法与 host 端启动配置；
2. **不做** clamp——CUDA 版没有 `MAX_EXP_*` 宏，SYCL 版同样不加
   （原因见“为什么 Swish 不需要溢出保护”）；
3. 复刻 Swish 特有的两个辅助函数 `swish`（FP32）与 `swish_half`（FP16），
   并**照搬 FP16 版的除序**：CUDA 的 `swish_half` 写成
   `__hmul(x, __hdiv(1, 1 + hexp(-x)))`，与 FP32 版的 `x / (1 + exp(-x))` 除序相反、
   数学等价。写成乘法是因为 half 的除法内部是一段较长的指令序列，
   反正都要算一次 `1 / d`，不如用乘法把结果乘回去；
4. **原样保留** CUDA 版 FP16 在 `x <= -11.09` 时饱和成 `-0.0`、
   FP32 极小负数下溢成 `-0.0`，以及向量化版本缺少 tail 分支等已知缺陷，
   仅在本文档与源码注释中照实标注；
5. 测试脚本的官方对照**照搬 CUDA 版**：使用脚本自己用
   `sigmoid + mul_` 两个算子拼出来的 `torch_swish`。数学上等价于
   `torch.nn.functional.silu`，但执行上是两个 elementwise kernel、
   中间多读写一遍显存，所以 `out_*_th` 那一栏明显偏慢——本次保持脚本原样不替换；
6. 打印格式沿用 LeetOneAPI 现有模块（CUDA 版 `swish.py` 用 `f"{v:<12}"` 补齐，
   SYCL 版去掉补齐，与 elementwise / relu / sigmoid 一致）；
7. 按已有模块的组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/swish/swish.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `swish.cu`，代码内含详细注释） |
| `kernels/swish/swish.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/swish/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/kernels/swish/README.md` | 本文件，模块设计说明 |

## CUDA → SYCL 概念对照

| CUDA | oneAPI / SYCL | 说明 |
| --- | --- | --- |
| `nvcc` 编译 `.cu` | `icpx -fsycl` 编译 `.sycl` | Intel oneAPI DPC++/C++ 编译器 |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` | 提交 kernel 的方式 |
| `grid` / `block` | `nd_range<1>` 的 global / local range | 总 work-item 数 = grid × block |
| 全局线程编号 `blockIdx.x*blockDim.x+threadIdx.x` | `item.get_global_id(0)` | 最常用的“我是第几个 work-item” |
| `float4` / `half2` | `sycl::vec<float,4>` / `sycl::vec<sycl::half,2>` | SIMD 向量类型 |
| `half` | `sycl::half` | FP16 类型，2 字节 |
| `expf` / `hexp` | `sycl::exp`（含 `sycl::half` 重载） | 指数函数 |
| `__hdiv` / `__hadd` / `__hmul` / `__hneg` | `sycl::half` 的 `/` / `+` / `*` 与一元 `-` | half 四则运算，SYCL 已定义运算符重载 |
| `__float2half(1.0f)` | `sycl::half(1.0f)` | 常数转 half |
| `LDST128BITS(value)`（`reinterpret_cast<float4*>`） | `sycl::vec<sycl::half,8>` 的 load/store 宏 | 128 位整体访存 |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构；本示例默认置空走 JIT(`spir64`) |

## Swish 曲线与数学性质

```text
  y = x / (1 + e^(-x)) = x * sigmoid(x)

3.0 ┤                          ***
    ┤                       ***
2.0 ┤                    ***
    ┤                 ***
1.0 ┤              ***
    ┤           ***
0.0 ┤*******+
    ┤      ***
-0.28┤   *
    └──────────────────────────────→ x
    -3   -2   -1   0   1   2   3

采样值：-3→-0.1423、-2→-0.2384、-1.279→-0.2785、-1→-0.2689、
       0→0、1→0.7311、2→1.7616、3→2.8577、4→3.9281
```

读图要点：

- **正半轴近似恒等映射**：`x` 稍大（约 3 以上）后 `sigmoid(x) ≈ 1`，
  `swish(x) ≈ x`，与 ReLU / ELU / GELU 的正半轴一致；
- **负半轴不是硬零**：`x < 0` 时输出是小负数，`x → -∞` 时趋近 0
  （因为 `x * 0` 里 `sigmoid(x)` 衰减得比 `x` 增长更快）；
  最小值约 `-0.2785`（出现在 `x ≈ -1.279`），所以 Swish **不是单调函数**；
- **它是“自门控”的**：`sigmoid(x)` 扮演门控——`x` 为正时门开（信号通过），
  `x` 为负时门关（信号被压制）。这就是 Swish 与 GELU 被称为
  self-gated / 平滑 ReLU 家族的原因；
- **在 `x = 0` 处光滑可导**：
  `swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`，
  在 `x = 0` 处等于 0.5（导数不为 0，这与 ReLU 的次梯度取 0 不同）；
- **与 GELU 的关系**：两者形状很接近（先下凹再单调上升），GELU 可以看成
  “用正态 CDF 当门控”，Swish 用的是 sigmoid 门控；Swish 的凹陷更深
  （-0.2785 对 GELU 的 -0.1700），但计算更便宜（没有 tanh 里的三次多项式）。

## 为什么 Swish 不需要溢出保护（与 gelu 的关键差异）

本模块的公式里有除法 `x / (1 + exp(-x))`，但两个极端都对应正确的数学极限：

| 输入 | `exp(-x)` | 结果 | 说明 |
| --- | --- | --- | --- |
| `x` 很大的正数（88） | `exp(-88) ≈ 6e-39` | `x / 1 = x` | `sigmoid(88) ≈ 1` |
| `x` 很大的负数（-88） | `exp(88)` 溢出为 `inf` | `x / inf = -0` | 数学极限就是 `x * 0` |
| `x = 0` | `exp(0) = 1` | `0 / 2 = 0` | 恰好过原点 |
| `x = ±inf` | `0 / inf` | `±inf` / `NaN` | 与 PyTorch 行为一致 |
| `x = NaN` | `NaN` | `NaN` | NaN 正常传播 |

关键在第二行：`exp(-x)` 溢出得到的 `inf` 恰好对应 `sigmoid(-∞) = 0`，
而 `x / inf = 0` 正是极限值，所以这里的溢出是“无害”的；
不像 gelu 那样会算出 `inf / inf = NaN`。

> 这是一个很好的横向对照：**同样的 `exp`，放在分母上安全（swish）、
> 只在负半轴出现也安全（elu），放进“自己除自己”的形式就不安全（gelu）。**

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。

### 0. 两个辅助函数：`swish` 与 `swish_half`

```cpp
// FP32：直接写成除法。x * (1 / (1 + exp(-x))) 与 x / (1 + exp(-x)) 等价，
// 但后者只做一次除法、不需要额外的乘法与中间变量，寄存器压力更小
static inline float swish(float x) {
  return x / (1.0f + sycl::exp(-x));
}

// FP16：half 没有隐式类型提升，常数要显式转成 sycl::half；
// 注意这里的除序与 FP32 版相反（先算 1 / d 再乘），与 CUDA 版一致
static inline sycl::half swish_half(sycl::half x) {
  return x * (sycl::half(1.0f) / (sycl::half(1.0f) + sycl::exp(-x)));
}
```

命名保持 CUDA 版原样（`swish` / `swish_half`），与 elu 模块的 `elu` / `elu_half` 一致。

### 1. `swish_f32`（FP32 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = swish(x[idx]);      // x / (1 + exp(-x))，等价于 x * sigmoid(x)
}
```

- 一个 work-item 只算一个元素，正确性最直观，是后面所有版本的基准；
- `if (idx < N)` 处理 `N` 不能被 work-group 覆盖时“多启动”的 work-item。

### 2. `swish_f32x4`（FP32 向量化版）

```cpp
int idx = 4 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  float4_vec reg_x = LOAD_FLOAT4(x + idx);   // 一次读 16 字节
  float4_vec reg_y;
  reg_y[0] = swish(reg_x[0]);
  reg_y[1] = swish(reg_x[1]);
  reg_y[2] = swish(reg_x[2]);
  reg_y[3] = swish(reg_x[3]);
  STORE_FLOAT4(y + idx, reg_y);              // 一次写 16 字节
}
```

- 与 elementwise 的 `f32x4` 不同，Swish 是**单输入**算子，没有 `b` 侧的第二路 load；
- 与 sigmoid 的 `f32x4` 相比，这里少了“先 clamp”的四行，多了一次乘法
  （`x * sigmoid(x)`），指令数量相当；
- “向量化”省的只是访存指令：每个分量都要算一次 `exp` 和一次除法，
  属于计算偏重的算子，收益有限；
- 读取写在 `if (idx < N)` **内部**（照搬 CUDA 版 `swish_f32x4_kernel`）。

### 3. `swish_f16`（FP16 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = swish_half(x[idx]);
}
```

- FP16 下有两处需要留意：
  1. **分母“吞掉小量”**：当 `exp(-x) < 2^-11`（约 4.9e-4）时，
     `1 + exp(-x)` 会因为 half 的加法舍入直接变成 1.0，于是 `sigmoid(x)` 被舍入成 1，
     `swish(x)` 退化成 `x`（例如 `x = 11` 时返回 11.0，与 PyTorch 一致）；
  2. **负方向饱和**：`x <= -11.09` 时 `exp(-x)` 溢出成 `inf`，
     `1 / inf = 0`，结果是 `x * 0 = -0.0`；而 PyTorch（内部用 float 算 sigmoid）
     会给出 `-7.37e-05` 这样的小值。绝对误差只有 7e-5，通常无感。

### 4. `swish_f16x2`（FP16 每 work-item 2 个元素）

```cpp
int idx = 2 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  half2_vec reg_x = LOAD_HALF2(x + idx);
  half2_vec reg_y;
  reg_y[0] = swish_half(reg_x[0]);
  reg_y[1] = swish_half(reg_x[1]);
  STORE_HALF2(y + idx, reg_y);
}
```

- **与 relu 的 f16x2 不同**：ReLU 有成对指令 `__hmax2`；Swish 既没有 `hexp2`
  也没有成对的除法，只能逐分量调用 `swish_half`。

### 5. `swish_f16x8`（FP16 每 work-item 8 个元素，unpack 写法）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half2_vec reg_x_0 = LOAD_HALF2(x + idx + 0);
half2_vec reg_x_1 = LOAD_HALF2(x + idx + 2);
half2_vec reg_x_2 = LOAD_HALF2(x + idx + 4);
half2_vec reg_x_3 = LOAD_HALF2(x + idx + 6);
// 8 个分量逐个算 Swish（4 个 half2 × 2 个分量）
...
```

- 每个 work-item 处理 8 个 half，拆成 4 个 `vec<half,2>`（`idx+0/2/4/6`）；
- “unpack” 指不整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 4 次读取是无条件执行的，越界判断只出现在写回处。

### 6. `swish_f16x8_pack`（128 位打包版）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half8_vec pack_x = LOAD_HALF8(x + idx);   // 一次读 128 位
half8_vec pack_y;
#pragma unroll
for (int i = 0; i < 8; ++i) {
  pack_y[i] = swish_half(pack_x[i]);      // 逐个 half，没有成对指令可用
}
if ((idx + 7) < N) {
  STORE_HALF8(y + idx, pack_y);           // 一次写 128 位
}
```

- 8 个 half 恰好 16 字节 = 128 位，一次 load/store 完成搬运；
- 循环写成 `++i` 逐个 half 算——Swish 没有成对指令，
  这一点与 sigmoid 的 pack 版写法相同（对照 `relu` 的 `i += 2`）。

## 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

### 1. FP16 在 `x <= -11.09` 时饱和成 `-0.0`

`exp(-x)` 在 `x <= -11.09` 时溢出成 `inf`，于是 `1 / inf = 0`，结果为 `x * 0 = -0.0`。
PyTorch 内部用 float 算 sigmoid，会给出 `-7.37e-05` 这样的小值——
绝对误差只有 `7e-5`，在 half 精度下通常无感，但**符号是 0 与负数的差别**。
实测（`swish_f16`，输入为 half 标量）：

| 输入 `x` | kernel 输出 | `torch.nn.functional.silu` |
| ---: | ---: | ---: |
| -9 | -0.0011100769 | -0.0011105513 |
| -11 | -0.0001835823 | -0.0001837156 |
| -11.09 | **-0.0** | -0.0001687009 |
| -12 | **-0.0** | -0.0000737301 |

即 `x = -11` 时两者还一致，从 -11.09 起 kernel 直接给出 `-0.0`。

### 2. FP32 的极小负数会被下溢成 `-0.0`

FP32 版本要到 `x` 非常小（`exp(-x)` 溢出、`x / inf`）时才会出现同样的情况，
阈值约在 `x ≈ -88` 附近；`torch.randn` 生成的输入不会触发。

### 3. 向量化版本的尾部（tail）分支缺失

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `swish_f32x4` | `idx < N` | 越界写 1~3 个 float |
| `swish_f16x2` | `idx < N` | 越界写 1 个 half |
| `swish_f16x8` | `(idx + 0/2/4/6) < N` | 越界写 1 个 half |
| `swish_f16x8_pack` | `(idx + 7) < N` | 末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S` / `K` 都取 1024 的倍数，启动的 work-item 恰好整段对齐，
不会暴露该问题。

### 4. `torch_swish` 对照实现是“两个 kernel”而不是一个

`torch_swish` 写成 `x * torch.sigmoid(x)`，PyTorch 会调度两个 elementwise kernel
（sigmoid 一次、mul 一次），中间结果还要多写读一遍显存；
公平的对照应该是 `torch.nn.functional.silu`（单个融合 kernel）。
这是 CUDA 版原本的写法，本次保留，仅在文档中说明。

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

> 与 CUDA 版的差异：CUDA 的 `TORCH_BINDING_SWISH` 只判断了 `(K / n_elements) <= 1024`，
> SYCL 版沿用前面几个模块的 3 项检查（大于 0、不超过 1024、能整除）。

各版本的 work-group 大小（`local`）：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每个 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `swish_f32` | 1 | 256 | 256 |
| `swish_f32x4` | 4 | 64 | 256 |
| `swish_f16` | 1 | 256 | 256 |
| `swish_f16x2` | 2 | 128 | 256 |
| `swish_f16x8` | 8 | 32 | 256 |
| `swish_f16x8_pack` | 8 | 32 | 256 |

## PyTorch 绑定

`swish.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python：

1. 用宏检查 `x` / `y` 的 dtype 与 XPU 设备，并检查两者形状一致；
2. 根据维度/形状计算 nd_range 的 global / local；
3. 用 `data_ptr()` 拿裸指针；从当前 XPU stream 取 `sycl::queue` 提交 kernel lambda。

调用链（以 `swish_f32` 为例）：

```text
swish.py
  └─ lib.swish_f32(x, y)                      # Python 调用（pybind11）
      └─ swish_f32(x, y)                      # C++ host 包装函数（宏生成）
          └─ submit_swish_f32(queue, ...)     # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)   # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

## 基准测试脚本

`swish.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `swish.sycl`；
2. 打印扩展构建目录与 XPU 设备名；
3. 对 `S` / `K` 的 9 种组合生成随机张量（FP32 与 FP16 各一套）；
4. `run_benchmark` 先 warmup，再运行 1000 次取平均；
5. 依次对比 6 个自定义 kernel 与 `torch_swish` 的正确性和耗时。

官方对照照搬 CUDA 版，保留 `out` 参数（结果原地写进 `y`）：

```python
def torch_swish(x, out=None):
    if out is None:
        return x * torch.sigmoid(x)
    else:
        torch.mul(x, torch.sigmoid(x), out=out)
        return out
```

它是两个 elementwise kernel 的组合（见“当前边界约束”第 4 条），
所以 `out_*_th` 明显偏慢；公平对照应该是 `torch.nn.functional.silu`。

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

`swish.py` 在 `load(...)` 前执行 `os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`，
默认只生成 `-fsycl-targets=spir64`（JIT），不需要 `ocloc`；
若已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 与 CUDA 版的对照：CUDA 版脚本用了 `--use_fast_math`（Swish 的开销主要在 `exp` 上，
> 收益明显）；SYCL 侧默认用精度完整的 `sycl::exp`，如需对齐可改用 `sycl::native::exp`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 swish.py`，结果为：

- FP32 三个版本（`swish_f32` / `swish_f32x4` / `torch_swish`）在同一份输入上一致；
- FP16 各版本彼此一致；FP16 与 FP32 之间存在约 `1/1024` 量级的量化差异；
- 负半轴输出是小负数（不是硬零），`x` 很负时结果趋近 `-0.0`；
- 没有出现 `NaN` / `inf`——这正是“`exp` 放在分母上安全”的体现。

完整输出见 `kernels/swish/README.md`。

### Arc A770 实测（S=4096, K=4096，JIT/spir64，一次运行结果）

Swish 是单输入算子：每个元素读 1 份、写 1 份，等效访存量 = `2 × N × sizeof(dtype)`；
本组 `N = 16,777,216`，FP32 为 134.2 MB，FP16 为 67.1 MB。

| 版本 | 耗时 (ms) | 等效带宽 |
| --- | ---: | ---: |
| f32 | 0.3222 | ~416 GB/s |
| f32x4 | 0.3270 | ~410 GB/s |
| torch f32（对照） | 0.8211 | ~164 GB/s |
| f16 | 0.2909 | ~231 GB/s |
| f16x2 | 0.1713 | ~392 GB/s |
| f16x8 | 0.3254 | ~206 GB/s |
| f16x8_pack | 0.1751 | ~383 GB/s |
| torch f16（对照） | 0.4374 | ~153 GB/s |

由此得到的结论：

1. **Swish 同样是彻底的访存带宽瓶颈**：FP32 三个版本都停在 ~410~416 GB/s，
   与 elementwise / relu / sigmoid / elu 同量级；
2. **五个算子的横向对照（同一台 A770、同一尺寸）**：
   relu `f32` 0.3221 ms、elu 0.3225 ms、swish 0.3222 ms、sigmoid 0.3222 ms 几乎完全相同，
   只有 gelu（0.3313 ms，算术强度最高）略慢约 3%——这说明当带宽打满后，
   一次 `exp` 甚至一次 `exp` + 除法都是“免费”的；
3. FP16 最快版本比 FP32 标量快约 **1.9 倍**（0.3222 ms → 0.1713 ms）；
4. `f16x8`（unpack）依然最慢（0.3254 ms，~206 GB/s），这是本仓库里反复出现的结论；
5. 对照实现 `torch_swish` 只有 ~164 GB/s（FP32）/ ~153 GB/s（FP16），
   因为它是 `sigmoid + mul_` 两个 kernel、中间多读写一遍显存，
   不是公平对比（应用 `torch.nn.functional.silu`）；
6. 注意上面这组数据里的 FP16 结果都没有 NaN——`exp` 在分母上
   溢出只会得到 `-0.0`，不会像 gelu 那样 `inf/inf`。

## 后续可以尝试的优化方向

- 补上向量化版本的 tail 分支与 `f16x8_pack` 的尾部回退，使任意 `N` 都安全；
- 用 `torch.nn.functional.silu` 做公平对照，替换掉脚本自拼的 `torch_swish`；
- 用 `sycl::native::exp` 对比精度与速度（对应 CUDA 的 `--use_fast_math`）；
- 对比 `x / (1 + exp(-x))` 与 `x * sigmoid(x)`（先算 sigmoid 再乘）的实测差异，
  看哪个版本在这套软件栈上寄存器压力更小；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 对比不同 work-group 大小、不同向量宽度对带宽/吞吐的实测影响。
