# Pet安装

本文档包含了Pet及其依赖的安装（包括Pytorch）。

- Pet的介绍请参阅 [`README`](https://github.com/BUPT-PRIV/Pet-dev/blob/main/README.md)。

**环境要求:**

- NVIDIA GPU， Linux， Python3.6+。
- Pytorch 1.6x-1.8x （推荐Pytorch-1.8.1），多个python标准库及COCO API；安装这些依赖的介绍请参阅下文。
- CUDA 10.2， 11.x (推荐10.2 and 11.1)

**注意：**

- Pet已被证实在CUDA >=10.2和CuDNN 7.5.1中可用。
- 请确保Pytorch及Cuda版本的兼容性。

## Python

通过conda创建虚拟环境并激活：

```
conda create -n pet python=3.6 -y
conda activate pet
```

## Pytorch and torchvision

安装支持CUDA的Pytorch。

1.安装 Pytorch-1.8.1：

```
pip3 install torch==1.8.1 --user
```

2.安装 torchvision：

```
pip3 install torchvision==0.9.1 --user
```

## Pet

1.克隆Pet仓库：

```
git clone https://github.com/BUPT-PRIV/Pet-dev.git
```

2.安装 requirements.txt：

```
cd Pet-dev
pip3 install -r requirements.txt --user
```

3.设置 `pet`：

```
sh make.sh
```
