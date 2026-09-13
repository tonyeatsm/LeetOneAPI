<div align="center">
  <p align="center">
    <h2>📚 LeetOneAPI: Modern SYCL/oneAPI Learn Notes with PyTorch for Beginners 🐑</h2>
  </p>
</div>

📚 **LeetOneAPI** is the **Intel oneAPI / SYCL** port of [**LeetCUDA**](https://github.com/xlite-dev/LeetCUDA). It rewrites LeetCUDA's teaching kernels with **SYCL/DPC++** (`icpx -fsycl`) and runs them on Intel GPUs through **PyTorch XPU** bindings. Every topic follows the same workflow as LeetCUDA: custom **SYCL kernel** implementation -> PyTorch **XPU bindings** -> run tests. 👉 TIPS: `✔️` = implemented; `/` = not supported; `❔` = TODO.

At the moment this repository only covers the [📚 Easy](#-easy-kernels) tier, i.e. the modules already ported to oneAPI:

- `elementwise`: `c[i] = a[i] + b[i]`, with FP32/FP16 scalar and vectorized variants;
- `histogram`: `y[v] = count(a[i] == v)`, an atomics-based counting kernel;
- `relu`: `y[i] = max(0, x[i])`, with FP32/FP16 scalar and vectorized variants;
- `elu`: `y[i] = x > 0 ? x : exp(x) - 1`, with FP32/FP16 scalar and vectorized variants;
- `gelu`: `y[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, the tanh approximation;
- `swish`: `y[i] = x / (1 + exp(-x))` (aka SiLU), with FP32/FP16 scalar and vectorized variants;
- `sigmoid`: `y[i] = 1 / (1 + exp(-x[i]))`, with FP32/FP16 scalar and vectorized variants.

The focus is not peak performance but understanding how the CUDA programming model maps onto SYCL/oneAPI, and how memory-access optimization ideas (vectorized loads, packing, FP16) carry over to Intel GPUs.

## 📖 Quick Start 🔥🔥

```bash
# 1. Clone the repository (replace with your own fork URL if needed)
git clone git@github.com:tonyeatsm/LeetOneAPI.git
cd LeetOneAPI

# 2. Start the Intel oneAPI container, mounting this repo into /workspace/LeetOneAPI
sudo docker run --device=/dev/dri -itd --name leetoneapi \
  -v /data/github-workspace/LeetOneAPI:/workspace/LeetOneAPI \
  --user root \
  -w /workspace/LeetOneAPI \
  intel/oneapi:2026.1.0-devel-ubuntu24.04

# 3. Enter the container (re-enter later with the same two commands)
sudo docker start leetoneapi
sudo docker exec -it leetoneapi /bin/bash
```

```bash
# Inside the container: check the oneAPI compiler and the visible Intel GPU
cd /workspace/LeetOneAPI
icpx --version
sycl-ls        # should list a level_zero:gpu device
```

```bash
# Install Python, venv and the XPU build of PyTorch
cd /workspace/LeetOneAPI

# (optional) proxy for apt/pip, if your network requires it
export HTTPS_PROXY=http://172.17.0.1:7897
export HTTP_PROXY=http://172.17.0.1:7897

apt update
apt install -y python3.12-dev python3.12-venv python3-pip

python3 -m venv /workspace/LeetOneAPI/.venv
source /workspace/LeetOneAPI/.venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ninja is required by the PyTorch SyclExtension build
python -m pip install ninja==1.13.2

# PyTorch XPU (Intel GPU backend)
python -m pip install torch==2.14.0 torchvision==0.29.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

```bash
# Verify the XPU device and run the kernels
cd /workspace/LeetOneAPI
source /workspace/LeetOneAPI/.venv/bin/activate

# Print the PyTorch version and the Intel GPU name
python3 -c "import torch; print(torch.__version__); print(torch.xpu.get_device_name(0))"

# Run elementwise (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/elementwise
python3 elementwise.py

# Run histogram (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/histogram
python3 histogram.py

# Run sigmoid (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/sigmoid
python3 sigmoid.py

# Run relu (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/relu
python3 relu.py

# Run elu (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/elu
python3 elu.py

# Run gelu (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/gelu
python3 gelu.py

# Run swish (.sycl is compiled on the first run, which takes a while)
cd /workspace/LeetOneAPI/kernels/swish
python3 swish.py
```

See [scripts/README.md](./scripts/README.md) for the same steps in a copy-paste friendly form.

## 📖 Contents

- [📖 Quick Start 🔥🔥](#-quick-start-)
- [📖 Contents](#-contents)
- [📖 Easy Kernels](#-easy-kernels)
- [📖 Environment & Build](#-environment--build)
- [📖 CUDA → SYCL Cheatsheet](#-cuda--sycl-cheatsheet)
- [🎉 Acknowledgements](#-acknowledgements)

## 📖 Easy Kernels

The kernels listed here are the ones already ported from LeetCUDA to oneAPI/SYCL. Each name is a Python-callable binding generated in the corresponding `.sycl` file; the linked `docs/kernels/` directory holds the design notes. 👉 TIPS: `/` = not supported; `✔️` = implemented; `❔` = TODO.

### 📚 Easy ⭐️

| 📖 Kernel | 📖 Elem DType | 📖 Acc DType | 📖 Docs | 📖 Level |
|:---|:---|:---|:---|:---|
| ✔️ [elementwise_add_f32](./kernels/elementwise/elementwise.sycl)|f32|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_add_f32x4](./kernels/elementwise/elementwise.sycl)|f32|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_add_f16](./kernels/elementwise/elementwise.sycl)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_add_f16x2](./kernels/elementwise/elementwise.sycl)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_add_f16x8](./kernels/elementwise/elementwise.sycl)|f16|/|[link](./kernels/elementwise/)|⭐️|
| ✔️ [elementwise_add_f16x8_pack](./kernels/elementwise/elementwise.sycl)|f16|/|[link](./kernels/elementwise/)|⭐️⭐️|
| ✔️ [histogram_i32](./kernels/histogram/histogram.sycl)|i32|/|[link](./kernels/histogram/)|⭐️|
| ✔️ [histogram_i32x4](./kernels/histogram/histogram.sycl)|i32|/|[link](./kernels/histogram/)|⭐️|
| ✔️ [relu_f32](./kernels/relu/relu.sycl)|f32|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f32x4](./kernels/relu/relu.sycl)|f32|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16](./kernels/relu/relu.sycl)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x2](./kernels/relu/relu.sycl)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x8](./kernels/relu/relu.sycl)|f16|/|[link](./kernels/relu/)|⭐️|
| ✔️ [relu_f16x8_pack](./kernels/relu/relu.sycl)|f16|/|[link](./kernels/relu/)|⭐️⭐️|
| ✔️ [elu_f32](./kernels/elu/elu.sycl)|f32|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f32x4](./kernels/elu/elu.sycl)|f32|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16](./kernels/elu/elu.sycl)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x2](./kernels/elu/elu.sycl)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x8](./kernels/elu/elu.sycl)|f16|/|[link](./kernels/elu/)|⭐️|
| ✔️ [elu_f16x8_pack](./kernels/elu/elu.sycl)|f16|/|[link](./kernels/elu/)|⭐️⭐️|
| ✔️ [gelu_f32](./kernels/gelu/gelu.sycl)|f32|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f32x4](./kernels/gelu/gelu.sycl)|f32|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16](./kernels/gelu/gelu.sycl)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x2](./kernels/gelu/gelu.sycl)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x8](./kernels/gelu/gelu.sycl)|f16|/|[link](./kernels/gelu/)|⭐️|
| ✔️ [gelu_f16x8_pack](./kernels/gelu/gelu.sycl)|f16|/|[link](./kernels/gelu/)|⭐️⭐️|
| ✔️ [swish_f32](./kernels/swish/swish.sycl)|f32|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f32x4](./kernels/swish/swish.sycl)|f32|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16](./kernels/swish/swish.sycl)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x2](./kernels/swish/swish.sycl)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x8](./kernels/swish/swish.sycl)|f16|/|[link](./kernels/swish/)|⭐️|
| ✔️ [swish_f16x8_pack](./kernels/swish/swish.sycl)|f16|/|[link](./kernels/swish/)|⭐️⭐️|
| ✔️ [sigmoid_f32](./kernels/sigmoid/sigmoid.sycl)|f32|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f32x4](./kernels/sigmoid/sigmoid.sycl)|f32|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16](./kernels/sigmoid/sigmoid.sycl)|f16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x2](./kernels/sigmoid/sigmoid.sycl)|f16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x8](./kernels/sigmoid/sigmoid.sycl)|f16|/|[link](./kernels/sigmoid/)|⭐️|
| ✔️ [sigmoid_f16x8_pack](./kernels/sigmoid/sigmoid.sycl)|f16|/|[link](./kernels/sigmoid/)|⭐️⭐️|

Design notes: [docs/kernels/elementwise](./docs/kernels/elementwise/), [docs/kernels/histogram](./docs/kernels/histogram/), [docs/kernels/relu](./docs/kernels/relu/), [docs/kernels/sigmoid](./docs/kernels/sigmoid/), [docs/kernels/elu](./docs/kernels/elu/), [docs/kernels/gelu](./docs/kernels/gelu/) and [docs/kernels/swish](./docs/kernels/swish/).

## 📖 Environment & Build

The port is developed and tested against the following setup:

| Item | Value |
|:---|:---|
| Container image | `intel/oneapi:2026.1.0-devel-ubuntu24.04` |
| Device | Intel Arc A770 (16 GB) |
| PyTorch | `2.14.0+xpu` (`torch.xpu` backend) |
| Python | 3.12 |
| Compiler | `icpx -fsycl` via the PyTorch `SyclExtension` (C++20) |

The `.sycl` sources are compiled by `torch.utils.cpp_extension.load`, which hands them to `icpx`/DPC++ and links the SYCL/XPU runtime — the oneAPI equivalent of passing `.cu` to `nvcc`.

Intel GPU device code can be generated in two ways:

- **JIT (default)**: the scripts set `TORCH_XPU_ARCH_LIST=""` before `load(...)`, so PyTorch only emits `-fsycl-targets=spir64`. The Intel GPU driver compiles it for the actual GPU at runtime. No `ocloc` is needed, any Intel GPU can run it; the only cost is the first-run compile/load overhead.
- **AOT**: if you have `ocloc` installed, export your architecture (e.g. `export TORCH_XPU_ARCH_LIST="xe-hpg"`) and PyTorch will add `-fsycl-targets=spir64_gen,spir64` and build device machine code offline.

## 📖 CUDA → SYCL Cheatsheet

| CUDA | oneAPI / SYCL |
|:---|:---|
| `nvcc` compiling `.cu` | `icpx -fsycl` compiling `.sycl` |
| `kernel<<<grid, block>>>(...)` | `queue.submit` + `parallel_for(nd_range)` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `item.get_global_id(0)` |
| `blockIdx.x` / `blockDim.x` / `threadIdx.x` | `item.get_group(0)` / `item.get_local_range(0)` / `item.get_local_id(0)` |
| `float4` / `half2` / `int4` | `sycl::vec<float,4>` / `sycl::vec<sycl::half,2>` / `sycl::vec<int,4>` |
| `atomicAdd(&y[v], 1)` | `sycl::atomic_ref<int, ...>(y[v]).fetch_add(1)` |
| `torch.cuda` / `.cuda()` / CUDA stream | `torch.xpu` / `.xpu()` / `c10::xpu` current XPU stream |
| `TORCH_CUDA_ARCH_LIST` | `TORCH_XPU_ARCH_LIST` |

Both `docs/kernels/elementwise/README.md` and `docs/kernels/histogram/README.md` explain the mapping in more depth.

## 🎉 Acknowledgements

This project is a study port built on top of [**LeetCUDA**](https://github.com/xlite-dev/LeetCUDA) — *Modern CUDA Learn Notes with PyTorch for Beginners*. The algorithms, kernel design and the "Easy -> Hard" teaching structure all come from LeetCUDA; this repository only re-expresses the Easy tier with SYCL/DPC++ and PyTorch XPU. Many thanks to the LeetCUDA author **DefTruth** and its contributors.
