# ReLU（修正线性单元）oneAPI / SYCL 移植设计说明

## 模块目标

本模块把 LeetCUDA 的 relu（修正线性单元）教学示例移植到 Intel oneAPI：

```text
y[i] = max(0, x[i]),  i = 0 .. N-1
```

在 CUDA 版中，同一个运算有 6 种写法，用于演示 **标量版 / 向量化 / FP16 / 128 位打包访存**
等带宽与访存优化手段；本 oneAPI 版本用 SYCL/DPC++ 复刻同样的 6 种写法：

1. `relu_f32`：FP32 标量版（每 work-item 1 个元素）
2. `relu_f32x4`：FP32 向量化版（每 work-item 4 个元素，16 字节访存）
3. `relu_f16`：FP16 标量版（`sycl::half`）
4. `relu_f16x2`：FP16 每 work-item 2 个元素（4 字节访存）
5. `relu_f16x8`：FP16 每 work-item 8 个元素（4 次 4 字节访存，unpack 写法）
6. `relu_f16x8_pack`：FP16 每 work-item 8 个元素（128 位打包访存）

> 命名说明：上面 6 个名字是 Python 侧可直接调用的绑定函数（与 CUDA 版一一对应）。
> SYCL 没有独立的 `__global__` 函数，这些计算逻辑实际位于源码
> `submit_relu_*` 函数里 `queue.submit + parallel_for` 提交的 kernel lambda 中；
> 下文为便于教学对照，仍按 6 个“kernel 版本”讲述。

**与 elementwise / sigmoid 的关系**：本模块与它们是一组姊妹示例，6 个版本一一对应，
区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` 换成 `y[i] = max(0, x[i])`。
前面模块讲过的线程编号、向量化访存、FP16 打包等知识本模块不再重复。

**本模块的独特价值**：ReLU 是**计算极轻**的算子——一次取最大值即可完成，没有 `exp`、
没有除法、也不会溢出。因此本模块除了教学，还承担一个“**性能上限参照**”的角色：
当计算量小到几乎可以忽略时，kernel 时间基本由访存主导，此时各种访存写法
（标量 / 向量化 / FP16 / 128 位打包）的差距会表现得最干净。

与 sigmoid 的三点关键差异：

| 对比项 | sigmoid | relu |
| --- | --- | --- |
| 公式 | `1 / (1 + exp(-x))` | `max(0, x)` |
| 值域 | `(0, 1)`，两端饱和 | `[0, +∞)`，正半轴不饱和 |
| 每元素计算 | 一次 `exp` + 一次加法 + 一次除法 | 一次取最大值 |
| 溢出保护 | 必须 `clamp` 后再 `exp` | 不需要 |
| 成对指令 | 无 `hexp2`，只能逐分量算 | 有 `__hmax2`，一次算 2 个 half |
| half 常数 | 需要预存 `const half f = __float2half(1.0f)` | 不需要（`0` 是字面量） |

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型如何映射到
SYCL/oneAPI，以及访存优化（向量化、FP16、128 位打包）如何跨硬件迁移。

## 本次移植范围

本模块严格照搬 LeetCUDA `kernels/relu/relu.cu` 的行为，**只做 CUDA → SYCL
的等价移植，不改算法逻辑**，具体包括：

1. 复刻 6 个 kernel 的写法（标量 / `float4` / `half` / `half2` / unpack / pack）
   与 host 端启动配置；
2. **不做** sigmoid 那样的 `clamp` / 溢出保护——ReLU 的值域天然落在 `[0, +∞)`，
   不产生新的数值，所以 CUDA 版没有 `MAX_EXP_*` 宏，SYCL 版同样不加
   （原因见“为什么 ReLU 不需要溢出保护”一节）；
3. **原样保留** CUDA 版 `f32x4` / `f16x2` / `f16x8` 缺少 tail 分支（只判断段首
   `(idx + 0) < N`）以及 `f16x8_pack` 尾部整段漏算的已知缺陷，仅在本文档与源码注释中
   照实标注（详见“当前边界约束”一节）；
4. 测试脚本的官方对照**照搬 CUDA 版**：直接调用 `torch.relu`，不像 `sigmoid.py`
   那样用 `partial(torch.sigmoid, out=y)`。因此 `out_f32_th` / `out_f16_th` 的耗时里
   包含每次迭代新建输出张量的开销，与 CUDA 版口径一致；
5. 打印格式沿用 LeetOneAPI 现有模块（elementwise / sigmoid）：
   `[1.42392182, 0.65998262], time:0.32214594ms Aha!`，
   不再使用 CUDA 版 `f"{v:<12}"` 的引号与左对齐补空格；
6. 按已有 elementwise / histogram / sigmoid 模块的文档、脚本、代码组织方式补充教学注释。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/relu/relu.sycl` | SYCL kernel + PyTorch XPU 绑定（对应 CUDA 版 `relu.cu`，代码内含详细注释） |
| `kernels/relu/relu.py` | 用 `torch.utils.cpp_extension.load` 编译 `.sycl` 并执行基准测试 |
| `kernels/relu/README.md` | 模块使用说明与测试输出 |
| `scripts/README.md` | oneAPI 容器与运行命令说明 |
| `docs/relu/README.md` | 本文件，模块设计说明 |

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
| `half` / 128 位打包用的 `float4` | `sycl::half` / `sycl::vec<sycl::half,8>` | FP16 类型与 16 字节打包向量 |
| `fmaxf(0.0f, x)` | `sycl::fmax(0.0f, x)` | FP32 取最大值 |
| `__hmax(__float2half(0.0f), x)` | `sycl::fmax(sycl::half(0.0f), x)` | FP16 标量取最大值，必须用 half 重载 |
| `__hmax2(a, b)` | `sycl::fmax(a, b)`（`sycl::vec<sycl::half,2>` 重载） | FP16 **成对**取最大值，一次处理 2 个 half |
| `__float2half(0.0f)` | `sycl::half(0.0f)` | 把常数 0 转成 half（字面量，编译期折叠） |
| `LDST128BITS(value)`（`reinterpret_cast<float4*>`） | `sycl::vec<sycl::half,8>` 的 load/store 宏 | 128 位整体访存 |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` 当前 XPU stream | PyTorch 设备 API 对应 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` | 目标 Intel GPU 架构列表；本示例默认置空走 JIT(`spir64`)，避免 devel 镜像缺 `ocloc` 时 AOT 失败 |

## ReLU 曲线与数学性质

```text
  y = max(0, x)

3.0 ┤                          ***
    ┤                       ***
2.0 ┤                    ***
    ┤                 ***
1.0 ┤              ***
    ┤           ***
0.0 ┤***********+
    └──────────────────────────────→ x
    -3     -2    -1    0    1    2    3

采样值：-3→0、-1→0、0→0、1→1、2→2、3→3
```

读图要点：

- **分段线性**：`x > 0` 时 `relu(x) = x`（恒等映射），`x <= 0` 时 `relu(x) = 0`；
  负半轴被“整流”成 0，这正是 Rectified 这个词的来历；
- **值域是 `[0, +∞)`，两端都不饱和**：与 sigmoid 的有界区间 `(0, 1)` 不同，
  正半轴导数**恒为 1**，梯度不会随 `x` 增大而衰减，所以深网络里比 sigmoid
  更不容易梯度消失——这是 ReLU 成为现代网络隐藏层默认选择的主因；
- **在 `x = 0` 处不可导**：左导数为 0、右导数为 1，工程上取次梯度（常用 0，
  PyTorch 的 `relu` 在 0 点也取 0）；
- **负半轴导数恒为 0**：若某个神经元的输出长期落在负半轴，它的梯度恒 0、
  权重不再更新，即所谓“**死亡 ReLU（dead ReLU）**”，这也是 leaky ReLU、
  GELU / SiLU 等变体出现的动机；
- **计算量极低**：整个算子只有一次比较/取最大值，没有 `exp`、没有除法，
  所以它既是很好的入门示例，也可以当作同类逐元素算子的性能上限参照。

## 为什么 ReLU 不需要溢出保护（与 sigmoid 的关键差异）

sigmoid 必须先 `clamp` 再 `exp`，否则 `exp(-x)` 的参数太大就会溢出成 `inf`
（见 `docs/sigmoid/README.md` 中的 `MAX_EXP_F32` / `MAX_EXP_F16` 说明）。
ReLU 完全没有这个问题：

| 输入情况 | `relu_f32_kernel` 的结果 |
| --- | --- |
| `x` 是很大的正数（如 `3.4e38`） | 原样返回，不会溢出（没有产生新的数值） |
| `x` 是很大的负数（如 `-3.4e38`） | 截断为 `0.0f` |
| `x` 是 `+inf` / `-inf` | `+inf` / `0.0f` |

只有一次比较和一次赋值，结果一定落在 `[0, +∞)` 内，因此本模块的源码里既没有
`fmin` / `fmax` 的“夹取”步骤，也没有 `MAX_EXP_*` 这类边界宏。
唯一需要留意的是 **NaN 语义**，见下文“当前边界约束”。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 1. `relu_f32`（FP32 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = sycl::fmax(0.0f, x[idx]);
}
```

- 一个 work-item 只算一个元素，正确性最直观，是后面所有版本的基准；
- `if (idx < N)` 处理 `N` 不能被 work-group 覆盖时“多启动”的 work-item；
- `sycl::fmax` 是 float 版本的取最大值函数，对应 CUDA 的 `fmaxf`
  （CUDA 里不能写成 `fmax`，那是 double 版本，会把参数提升成 double 再比较）；
- 参数写成 `sycl::fmax(0.0f, x[idx])` 只是习惯写法，`fmax` 本身是对称的
  （NaN 的情况除外，见“当前边界约束”）。

### 2. `relu_f32x4`（FP32 向量化版）

每个 work-item 连续处理 **4 个 float（16 字节）**：

```cpp
int idx = 4 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  float4_vec reg_x = LOAD_FLOAT4(x + idx);   // 一次读 16 字节
  float4_vec reg_y;
  reg_y[0] = sycl::fmax(0.0f, reg_x[0]);
  reg_y[1] = sycl::fmax(0.0f, reg_x[1]);
  reg_y[2] = sycl::fmax(0.0f, reg_x[2]);
  reg_y[3] = sycl::fmax(0.0f, reg_x[3]);
  STORE_FLOAT4(y + idx, reg_y);              // 一次写 16 字节
}
```

- 把一段连续内存 reinterpret 成 `sycl::vec<float,4>`（对应 CUDA 的 `float4`），
  用一次加载/存储完成 16 字节搬运，4 次 4 字节访存被压缩成 1 次 16 字节访存；
- 与 elementwise 不同，ReLU 和 sigmoid 一样是**单输入**算子：只需读 `x` 一份数据，
  没有 `b` 侧的第二路 load；
- 注意读取写在 `if (idx < N)` **内部**（照搬 CUDA 版 `relu_f32x4_kernel` 的写法；
  `sigmoid.sycl` 的向量化版本把读取放在判断之外，两者在这一点上不同）；
- ReLU 的计算只有 4 次取最大值，比 sigmoid 的 4 次 `exp` 轻得多，所以
  “省下的访存指令”更容易直接体现在耗时上。

### 3. `relu_f16`（FP16 标量版）

```cpp
int idx = static_cast<int>(item.get_global_id(0));
if (idx < N) {
  y[idx] = h_relu(x[idx]);   // = sycl::fmax(sycl::half(0.0f), x[idx])
}
```

- 数据类型为 `sycl::half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- 取最大值必须用 half 版本的 `sycl::fmax`，对应 CUDA 的 `__hmax`
  （直接比较 16 位 half）；若写成 float 版本，会先把 half 提升到 float、
  算完再转回来，多出两条转换指令；
- 与 sigmoid 不同，这里**不需要**预存 `const half f = __float2half(1.0f)`：
  ReLU 只需要一个常数 0，而 `sycl::half(0.0f)` 的参数是字面量，
  编译器会直接折叠成 half 常量，不产生运行时的类型转换开销。

### 4. `relu_f16x2`（FP16 每 work-item 2 个元素）

```cpp
int idx = 2 * static_cast<int>(item.get_global_id(0));
if (idx < N) {
  half2_vec reg_x = LOAD_HALF2(x + idx);
  half2_vec reg_y = LOAD_HALF2(y + idx);   // 冗余读取，照搬 CUDA 版
  reg_y[0] = h_relu(reg_x[0]);
  reg_y[1] = h_relu(reg_x[1]);
  STORE_HALF2(y + idx, reg_y);
}
```

- 一次用 `sycl::vec<sycl::half,2>`（2 个 half = 4 字节）完成两个元素的搬运；
- 代码里 `half2_vec reg_y = LOAD_HALF2(y + idx);` 先读了一次 `y`，但紧接着
  `reg_y[0]` / `reg_y[1]` 都会被覆盖，这次读取对结果没有影响，属于**冗余访存**
  （照搬 CUDA 版，本次只补充注释，不改逻辑）；
- ReLU 其实有成对指令（CUDA 的 `__hmax2` / SYCL 的 `sycl::fmax` 向量重载），
  但本 kernel 为了与 elementwise / sigmoid 的 `f16x2` 保持一致的“逐分量展开”写法，
  仍按两个分量分别调用标量 `h_relu`；成对指令的用法放在 Kernel 6。

### 5. `relu_f16x8`（FP16 每 work-item 8 个元素，unpack 写法）

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
- 这里的 “unpack” 指：数据在内存里本就是连续的 8 个 half，本 kernel
  不把它整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 相比 `f16x2`，每个 work-item 做更多工作，能摊薄索引计算等固定开销；
- 4 次读取是**无条件执行**的（越界判断只出现在写回处），照搬 CUDA 版。

### 6. `relu_f16x8_pack`（128 位打包版）

```cpp
int idx = 8 * static_cast<int>(item.get_global_id(0));
half8_vec pack_x = LOAD_HALF8(x + idx);   // 一次读 128 位
half8_vec pack_y;
#pragma unroll
for (int i = 0; i < 8; i += 2) {
  // 把 pack_x[i]、pack_x[i+1] 两个 half 看成一个 vec<half,2>，一次成对取最大值
  HALF2_AT(&pack_y[i]) = h_relu2(HALF2_AT(&pack_x[i]));
}
if ((idx + 7) < N) {
  STORE_HALF8(y + idx, pack_y);           // 一次写 128 位
}
```

- 8 个 half 恰好是 16 字节 = 128 位，用 `sycl::vec<sycl::half,8>` 一次 load/store，
  比 Kernel 5 的多条 4 字节访存指令更少；
- 与 elementwise / sigmoid 的 `f16x8_pack` 一样，这里的 “pack” 指的是把连续 8 个
  half 整体当作 128 位向量搬运；计算部分仍是按 half 粒度处理；
- **这里是本模块与 sigmoid 在写法上最有意思的差异**：
  sigmoid 没有成对的 `hexp2`，循环只能 `i++` 逐个 half 调用 `hexp`；
  而 ReLU 有成对指令（CUDA 的 `__hmax2`，SYCL 里是 `sycl::fmax` 的
  `vec<half,2>` 重载），一次就能处理一个 `half2`（2 个 half），
  所以循环步长是 `i += 2`，8 个元素只需 4 次成对操作——这正是 CUDA 版
  `relu_f16x8_pack_kernel` 与 `sigmoid_f16x8_pack_kernel` 在循环步长上的区别
  （是否最终合成一条成对指令由后端决定，写法与语义和 CUDA 版一致）；
- CUDA 版用局部数组 `half pack_x[8]` + `LDST128BITS`（reinterpret 成 `float4`）
  完成 128 位搬运；SYCL 版直接使用 `sycl::vec<sycl::half,8>`，语义相同。
  注意 `HALF2_AT(&pack_x[i])` 对向量元素取地址，编译器同样可能把它落到私有内存，
  配合 `#pragma unroll` 与编译期常数下标才有机会优化进寄存器——这一点与 CUDA 版
  的局部数组完全对应。

## 当前边界约束（照搬 CUDA 版，已知缺陷，本模块不修复）

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部（tail）标量回退分支；本模块的向量化版本**两个都没有**，
越界判断只检查了段的起始下标：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `relu_f32x4` | `(idx + 0) < N` | 最后一个 work-group 的部分 work-item 会把 `idx+1..idx+3` 写到 `y` 合法范围之外（越界写 1~3 个 float） |
| `relu_f16x2` | `(idx + 0) < N` | 同上，越界写 1 个 half |
| `relu_f16x8` | `(idx + 0/2/4/6) < N` | 同上，越界写 1 个 half |
| `relu_f16x8_pack` | `(idx + 7) < N` | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S` / `K` 都取 1024 的倍数（元素数是 256 的倍数），
启动的 work-item 恰好整段对齐，因此不会暴露该问题。

本次只做移植、不改逻辑，所以该风险仅在本文档和源码注释中写明。
如需在生产中使用，应参照 elementwise 补充尾部处理：

```cpp
if ((idx + 3) < N) {
  // 4 个元素都有效，走向量化分支
} else if (idx < N) {
  // 尾部不足 4 个元素，退回逐元素处理
}
```

### NaN 语义差异（已识别，本次不修复）

本模块 kernel 用 `sycl::fmax`（对应 CUDA 的 `fmaxf` / `__hmax`）取最大值，
它们遵循 IEEE 754-2008 的 `maxNum` 语义：**一个参数是 NaN 时返回另一个参数**。
因此 `relu_f32` 遇到 `x = NaN` 会输出 `0.0f`，而 PyTorch 的 `torch.relu`
会把 NaN 传播到输出。两者在含 NaN 的输入上结果不同。

本模块的基准测试用 `torch.randn` 生成有限值输入，不会触发该差异；
按“照搬 CUDA 版、不改逻辑”的原则，本次不修改 kernel。

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

> 与 CUDA 版的差异：CUDA 的 `TORCH_BINDING_RELU` 只判断了 `(K / n_elements) <= 1024`，
> 没有 `K % n_elements == 0` 与 `> 0` 这两项检查。若 `K` 不能被向量宽度整除，
> CUDA 版会按“一行一个 block”启动，导致每行末尾几个元素永远不被处理（静默漏算）。
> SYCL 移植版本沿用 elementwise / sigmoid 模块的 3 项检查，这种情况回退到展平启动；
> 本模块基准的形状都能整除，两种写法的启动配置完全一致。

各版本的 work-group 大小（`local`）如下：

| 版本 | 每 work-item 元素数 | work-group 大小（local） | 每个 work-group 处理元素数 |
| --- | ---: | ---: | ---: |
| `relu_f32` | 1 | 256 | 256 |
| `relu_f32x4` | 4 | 64 | 256 |
| `relu_f16` | 1 | 256 | 256 |
| `relu_f16x2` | 2 | 128 | 256 |
| `relu_f16x8` | 8 | 32 | 256 |
| `relu_f16x8_pack` | 8 | 32 | 256 |

## PyTorch 绑定

`relu.sycl` 用宏批量生成 6 个 host 函数，再通过 `PYBIND11_MODULE` 暴露给 Python，
与 CUDA 版一一对应。主要工作：

1. 用宏检查 `x` / `y` 的 dtype 与 XPU 设备，并检查两者形状一致；
2. 根据维度/形状计算 nd_range 的 global / local；
3. 用 `data_ptr()` 拿裸指针（XPU 张量内存即 SYCL USM 内存，可直接交给 kernel）；
4. 从当前 XPU stream 取 `sycl::queue`，`queue.submit` 提交 kernel lambda。

调用链（以 `relu_f32` 为例）：

```text
relu.py
  └─ lib.relu_f32(x, y)                       # Python 调用（pybind11）
      └─ relu_f32(x, y)                       # C++ host 包装函数（宏生成）
          └─ submit_relu_f32(queue, ...)      # SYCL host 启动函数
              └─ queue.submit + parallel_for(nd_range)   # SYCL 提交
                  └─ Intel GPU 硬件并行执行 kernel lambda
```

本模块与其它两个逐元素模块的 host 函数签名完全一致（都是单输入算子）：

| 模块 | host 函数签名 | kernel 参数 |
| --- | --- | --- |
| elementwise | `elementwise_add_xxx(a, b, c)` | `a, b, c, N` |
| sigmoid | `sigmoid_xxx(x, y)` | `x, y, N` |
| relu | `relu_xxx(x, y)` | `x, y, N` |

## 基准测试脚本

`relu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `relu.sycl`（`.sycl` 源文件会被
   PyTorch 识别，交给 `icpx`/DPC++ 编译并链接 SYCL/XPU 运行库）；
2. 打印构建目录与 XPU 设备名（与 elementwise / histogram / sigmoid 对齐）；
3. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合在 XPU 上生成随机张量
   （FP32 与 FP16 各一套，显式 `.contiguous()`）；
4. `run_benchmark` 先 warmup，再运行 1000 次取平均；
5. 依次对比 6 个自定义 kernel 与 PyTorch 官方 `torch.relu` 的正确性和耗时。

与 CUDA 版 `relu.py` 的对应关系：

| CUDA 版 | SYCL 版 | 说明 |
| --- | --- | --- |
| `torch.randn((S, K)).cuda()` | `torch.randn((S, K)).xpu()` | 设备张量创建 |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` | 计时前的同步 |
| `torch.cuda.get_device_name()` | `torch.xpu.get_device_name(0)` | 设备名 |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST`（默认置空走 JIT） | 目标架构 |
| `f"{v:<12}"` 左对齐补齐输出 | 直接打印数值列表 | 打印格式，见上文“本次移植范围”第 5 条 |

官方对照的写法与 CUDA 版一致，直接调用 `torch.relu`：

```python
run_benchmark(lib.relu_f32, x, "f32", y)
run_benchmark(lib.relu_f32x4, x, "f32x4", y)
run_benchmark(torch.relu, x, "f32_th")      # 每次迭代返回新张量，不含 out 参数
...
run_benchmark(torch.relu, x_f16, "f16_th")
```

这与 `sigmoid.py` 的 `partial(torch.sigmoid, out=y)` 不同：`torch.relu` 会分配新的
输出张量，所以 `out_*_th` 的耗时里包含分配开销，数值上通常比自定义 kernel 慢——
这是 CUDA 版原本的口径，本模块照搬，便于与 LeetCUDA 的实测数据对照。

## 运行环境与编译方式（JIT / AOT）

当前环境为：

- 容器镜像：`intel/oneapi:2026.1.0-devel-ubuntu24.04`
- 设备：Intel Arc A770（16 GB）
- PyTorch：XPU 版本（`torch.xpu`，实测版本 2.14.0+xpu）
- 编译方式：PyTorch `SyclExtension` 识别 `.sycl` 文件，交给 `icpx -fsycl` 编译

与 elementwise / histogram / sigmoid 相同，`relu.py` 在 `load(...)` 前执行
`os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")`：

- 未显式设置时只生成 `-fsycl-targets=spir64`（JIT），由 Intel GPU 驱动即时编译；
- 不需要 `ocloc`，任何 Intel GPU 都能跑，缺点是首次运行有编译/加载开销；
- 若机器已装好 `ocloc` 且需要 AOT，可自行 `export TORCH_XPU_ARCH_LIST=<架构名>`。

> 另外注意：PyTorch 2.14 起要求 C++20，编译时不要用 `-std=c++17` 覆盖
> `SyclExtension` 自动添加的 `-std=c++20`。

## 运行结果与观察

在 `leetoneapi` 容器内运行 `python3 relu.py`，结果为：

- FP32 三个版本（`relu_f32` / `relu_f32x4` / `torch.relu`）在同一份输入上一致；
- FP16 各版本（`relu_f16` / `relu_f16x2` / `relu_f16x8` / `relu_f16x8_pack`）
  彼此一致；FP16 与 FP32 之间存在约 `1/1024` 量级的量化误差（half 精度损失），
  因为 ReLU 是“原样返回或清零”，误差不会被放大；
- 结果都落在 `[0, +∞)` 内，负输入被清成 `0`，没有出现 `NaN` / `inf`；
- 随机输入的前两个元素可能本来就是负数，被 ReLU 截成 `0`（输出 `'0.0'`），
  属于正常现象，判断正确性要看同组的 6 个版本与 `out_*_th` 是否一致。

完整输出见 `kernels/relu/README.md`。

### Arc A770 实测（S=4096, K=4096，JIT/spir64，一次运行结果）

ReLU 是单输入算子：每个元素读 1 份、写 1 份，所以等效访存量 = `2 × N × sizeof(dtype)`。
本组 `N = 4096 × 4096 = 16,777,216`，FP32 等效访存量 134.2 MB，FP16 为 67.1 MB。

| 版本 | 耗时 (ms) | 等效带宽 |
| --- | ---: | ---: |
| f32 | 0.3221 | ~417 GB/s |
| f32x4 | 0.3265 | ~411 GB/s |
| torch f32 | 0.3248 | ~413 GB/s |
| f16 | 0.2940 | ~228 GB/s |
| f16x2 | 0.1712 | ~392 GB/s |
| f16x8 | 0.3381 | ~199 GB/s |
| f16x8_pack | 0.1739 | ~386 GB/s |
| torch f16 | 0.1753 | ~383 GB/s |

由此得到的结论：

1. **ReLU 是彻底的访存带宽瓶颈**：FP32 三个版本都停在 ~411~417 GB/s，
   FP16 最快的版本 ~392 GB/s，与 elementwise / sigmoid 在同一台机器上测到的
   上限完全一致；
2. **本模块最值得看的一条对照**：把这张表和 `docs/sigmoid/README.md` 里
   同样 `S=4096, K=4096` 的实测数据并排看——sigmoid 每个元素要多算一次 `exp`
   和一次除法，但两组耗时几乎逐行相同（例如 `f32` 0.3221 vs 0.3222 ms、
   `f16x2` 0.1712 vs 0.1717 ms、`f16x8_pack` 0.1739 vs 0.1762 ms）。
   这说明在这个规模下 `exp` 的算术开销已经被访存完全掩盖：**算子的“贵”
   与“便宜”在带宽打满时看不出差别**，也印证了 ReLU 作为逐元素算子
   “性能上限参照”的定位；
3. FP16 最快版本比 FP32 快约 **1.9 倍**（0.3221 ms → 0.1712 ms），
   与“访存字节数减半”的预期一致；
4. FP16 内部差异明显：`f16x2`（~392 GB/s）与 `f16x8_pack`（~386 GB/s）最快，
   而 `f16x8`（unpack 写法，把 8 个元素拆成 4 次 4 字节访存）只有 ~199 GB/s，
   甚至略慢于 `f16` 标量版（~228 GB/s）——这与 elementwise / sigmoid 在
   Arc A770 上的结论一致，说明“多段 4 字节访存”在这套软件栈上不是好选择；
5. 向量化对 FP32 没有收益（`f32x4` 0.3265 ms ≈ `f32` 0.3221 ms ≈
   `torch f32` 0.3248 ms），符合“已经吃满带宽时，减少访存指令条数不再带来
   额外收益”的判断；
6. 小规模（如 `S=1024, K=1024`）时各版本差异变小、甚至出现 `f32` 比
   `f32x4` 慢的波动，因为 kernel 时间已经接近启动开销量级。

> 跨硬件对比：LeetCUDA 参考实现在 `S=4096, K=4096` 上 `f16x8pack` 约 `0.0147 ms`、
> `f32` 约 `0.1885 ms`，其中 FP16 换算出的等效带宽超过该卡理论显存带宽，
> 说明那组数字受 L2 缓存命中或计时方式影响。因此跨硬件只比趋势
> （FP16 快于 FP32、向量/打包版快于 unpack 版），不比绝对值。

## 后续可以尝试的优化方向

- 补上 `f32x4` / `f16x2` / `f16x8` 的 tail 分支与 `f16x8_pack` 的尾部回退，使任意
  `N` 都安全（对齐 LeetOneAPI elementwise 模块的写法）；
- 去掉 `relu_f16x2` 里对 `y` 的冗余读取；
- 用成对指令（`sycl::fmax` 的 `vec<half,2>` 重载，对应 `__hmax2`）重写
  `f16x2` / `f16x8`（unpack）版本，减少指令数；
- 引入 grid-stride loop，使固定大小的 nd_range 也能处理任意大 `N`；
- 对比不同 work-group 大小、不同向量宽度对带宽的实测影响（ReLU 计算极轻，
  是观察访存上限的最佳样本）；
- 与 `torch.relu` 对比时补充 `torch.nn.functional.relu`、`clamp_min(0)`
  等不同实现路径的性能差异；
- 安装含 `ocloc` 的工具链，对比 AOT 与 JIT 的差距。
