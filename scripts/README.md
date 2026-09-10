# Fork
```bash

# 注意：这里替换成你Fork后的仓库地址
git clone git@github.com:tonyeatsm/LeetOneAPI.git
cd LeetOneAPI

```
# 使用Docker挂载你的代码仓库
```bash

sudo docker run --device=/dev/dri -itd --name leetoneapi \
-v /data/github-workspace/LeetOneAPI:/workspace/LeetOneAPI \
--user root \
-w /workspace/LeetOneAPI \
intel/oneapi:2026.1.0-devel-ubuntu24.04

# Enter
sudo docker start leetoneapi 
sudo docker exec -it leetoneapi /bin/bash

# commit
sudo docker commit leetoneapi leetoneapi:20260909

# 在容器内初始化并开发
cd /workspace/LeetOneAPI
# oneAPI 编译环境：基础镜像默认已把 icpx 加入 PATH
icpx --version

# 确认 Intel GPU 是否对容器可见（应列出 level_zero:gpu 设备）
sycl-ls

```


# 安装Python
```bash

# Enter
sudo docker start leetoneapi 
sudo docker exec -it leetoneapi /bin/bash


# apt / pip 若需要代理，先设置代理，例如：
export HTTPS_PROXY=http://172.17.0.1:7897
export HTTP_PROXY=http://172.17.0.1:7897

cd /workspace/LeetOneAPI
apt update
apt install -y python3.12-dev python3.12-venv python3-pip
python3 --version

python3 -m venv /workspace/LeetOneAPI/.venv
source /workspace/LeetOneAPI/.venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ninja 是 PyTorch SyclExtension 编译必需
python -m pip install ninja==1.13.2

# PyTorch XPU 版本（Intel GPU 后端）
python -m pip install torch==2.14.0 torchvision==0.29.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu

```

# Quick Start
```bash
# Enter
sudo docker start leetoneapi 
sudo docker exec -it leetoneapi /bin/bash


# 进入工程并检查 XPU 设备（若 PyTorch XPU 尚未安装，跳到下一节“安装Python”）
cd /workspace/LeetOneAPI
python3 -c "import torch; print(torch.__version__); print(torch.xpu.get_device_name(0))"

```


# Easy
```bash
sudo docker start leetoneapi 
sudo docker exec -it leetoneapi /bin/bash
source /workspace/LeetOneAPI/.venv/bin/activate

# 打印 Intel GPU 型号与 XPU 可用性
cd /workspace/LeetOneAPI/kernels/elementwise
python3 -c "import torch; print('XPU =', torch.xpu.get_device_name(0)); print('xpu available =', torch.xpu.is_available())"

# 逐元素
cd /workspace/LeetOneAPI/kernels/elementwise
python3 elementwise.py

# 直方图统计
cd /workspace/LeetOneAPI/kernels/histogram
python3 histogram.py

```

