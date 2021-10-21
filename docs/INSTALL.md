# Installing Pet

This document covers how to install Pet, its dependencies (including Pytorch), and the COCO dataset.

- For general information about Pet, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python3.6+
- Pytorch-1.8.1 (recommended), various standard Python packages and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Pet has been tested extensively with CUDA >= 10.2 (recommended) and cuDNN 7.5.1.


## Pytorch and torchvision

Install Pytorch with CUDA support.

1. Install Pytorch-1.8.1:

```
pip3 install torch==1.8.1 --user
```

2. Install torchvision:

```
pip3 install torchvision==0.9.1 --user
```

## Pet

1. Clone the Pet repository:

```
git clone https://github.com/BUPT-PRIV/Pet-dev.git
```

2. Install the requirements.txt:

```
cd Pet-dev
pip3 install -r requirements.txt --user
```

3. Set up `pet`:

```
sh make.sh
```
