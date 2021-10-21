# 添加新的模块

Pet支持自定义新的模块与算法，请遵循Pet对应模块的规范。

Pet的模型构建过程请参考 [模型构建](../tutorials/model_building_zh.md)。本篇示例教程以添加一个骨干网络为例。

## 添加新的骨干网络

Pet的骨干网络定义在 `pet.lib.backbone `和 `pet.vision.modeling.backbone` 中，同时完整的骨干网络应支持所有任务类型，对应的输出格式为单个张量（分类任务）与特征列表（其他任务，可配合特征金字塔等特征增强方式）。

为方便教学与运行，本文仅在 `pet.vision.modeling.backbone` 中定义新的用于分类的backbone，网络仅输出一个维度与特征类别数量相同的张量。

### 网络结构

网络共分为两个模块，卷积模块与线性模块。卷积模块中，模型由多个卷积层堆叠，线性模块由多个线性层堆叠，模块内部每层均有激活函数，两个模块之间使用全局平均池化来进行维度统一。

为展示使用配置文件控制模型结构，

使用了4个卷积层，卷积核大小分别为7、5、3、1，并且均设置`stride=2`，总共16倍下采样。通过`AdaptiveAvgPool`模块，特征图维度变为`1 * 1 * 1024`，再通过三个全连接层，将输出长度变为10，对应Cifar10中每个类别的logit值。

网络结构如下所示：

|  名称   | 卷积核大小 | 输入特征图维度 | 输出特征图维度 |
| :-----: | :--------: | :------------: | :------------: |
|  Conv1  |    7*7     |       3        |       64       |
|  Conv2  |    5*5     |       64       |      128       |
|  Conv3  |    3*3     |      128       |      256       |
|  Conv4  |    1*1     |      256       |      1024      |
| AvgPool |            |      1024      |      1024      |
|   FC1   |            |      1024      |      512       |
|   FC2   |            |      512       |       32       |
|   FC3   |            |       32       |       10       |

网络层打印结果如下：

```
GeneralizedCNN(
  (backbone): ExampleNet(
    (conv_module): Sequential(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        (1): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (1): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(2, 2))
        (1): ReLU(inplace=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=1)
    (linear_module): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): ReLU6(inplace=True)
      (2): Linear(in_features=512, out_features=32, bias=True)
      (3): ReLU6(inplace=True)
      (4): Linear(in_features=32, out_features=10, bias=True)
    )
  )
  (global_cls): GlobalCls(
    (onehot): OneHotModule(
      (post_processor): OneHotPostProcessor()
    )
  )
)
```

## 配置文件

在`pet/lib/config/model/backbone`文件中，添加ExampleNet的默认配置文件：

```python
MODEL.EXAMPLENET = CfgNode()
MODEL.EXAMPLENET.FEATURES = [64, 128, 256, 1024] # 卷积层各层输出维度
MODEL.EXAMPLENET.KERNELS = [7, 5, 3, 1] # 卷积层各层卷积核大小
MODEL.EXAMPLENET.STRIDES = [2, 2, 2, 2] # 卷积层各层stride
MODEL.EXAMPLENET.PADDINGS = [3, 2, 1, 0] # 卷积层各层padding
MODEL.EXAMPLENET.CONV_ACT = ['ReLU', 'ReLU', 'ReLU', 'ReLU'] # 卷积层各层激活函数
MODEL.EXAMPLENET.LINEAR_ACT = 'ReLU6' # 线性层激活函数
MODEL.EXAMPLENET.FC_FEATURES = [512, 32] # 线性层隐层维度
```

在使用时，可以方便地通过修改配置文件来调整模型结构。修改方式可参考[配置系统](../tutorials/configs_zh.md)。

## 代码

在`pet/viison/modeling/backbone`中，新建`ExampleNet.py`文件。代码如下：

```python
import torch.nn as nn
import torch.nn.functional as F

from pet.lib.layers import make_conv, make_act
from pet.vision.modeling import registry


class ExampleNet(nn.Module):
    def __init__(self, cfg):
        super(ExampleNet, self).__init__()
        self.dim_in = 3
        self.spatial_in = [1]

        features = cfg.MODEL.EXAMPLENET.FEATURES
        kernel_size = cfg.MODEL.EXAMPLENET.KERNELS
        strides = cfg.MODEL.EXAMPLENET.STRIDES
        paddings = cfg.MODEL.EXAMPLENET.PADDINGS
        conv_act = cfg.MODEL.EXAMPLENET.CONV_ACT
        fc_act = cfg.MODEL.EXAMPLENET.LINEAR_ACT
        fc_features = cfg.MODEL.EXAMPLENET.FC_FEATURES
        num_classes = cfg.MODEL.ONEHOT.NUM_CLASSES

        features.insert(0, self.dim_in)  # 添加输入维度

        conv_layers = []
        for i, (kernel, stride, pad, act) in enumerate(zip(kernel_size, strides, paddings, conv_act)):
            conv_layers.append(
                make_conv(
                    in_channels=features[i],
                    out_channels=features[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=pad,
                    act=make_act(act)
                )
            )
        self.conv_module = nn.Sequential(*conv_layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        fc_features.insert(0, features[-1])
        fc_layers = []
        for i in range(len(fc_features)-1):
            fc_layers.append(nn.Linear(fc_features[i], fc_features[i+1]))
            fc_layers.append(make_act(fc_act))
        fc_layers.append(nn.Linear(fc_features[-1], num_classes))
        self.linear_module = nn.Sequential(*fc_layers)

        self.dim_out = [num_classes]
        self.spatial_out = [(1, 1)]

    def forward(self, x):
        """
        Args:
            x (tensor)

        Returns:
            x (tensor)
        """
        x = self.conv_module(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_module(x)
        return x

# ---------------------------------------------------------------------------- #
# ExampleNet
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("example_net")
def examplenet(cfg):
    """
    Args:
        cfg (CfgNode)

    Returns:
        nn.Module
    """
    model = ExampleNet(cfg)
    return model
```

**注意事项:**

- 网络的入参有cfg来结合配置文件修改模型结构。
- 完整的backbone应该有两种输出格式，分别用于分类任务和其他任务。
- 用registry注册该backbone，并设置好调用字段。

- 需要在`pet/viison/modeling/backbone/__init__.py`中导入该backbone

  ```python
  from .ExampleNet import *
  ```

## 使用

配置文件编写可以参考已有的Cifar10配置文件，需要修改 `MODEL.BACKBONE` 为` "example_net"`。

```yaml
MISC:
  CKPT: "ckpts/vision/Cifar/examplenet_cifar10"
  CUDNN: True
MODEL:
  BACKBONE: "example_net"
  GLOBAL_HEAD:
    CLS:
      ONEHOT_ON: True
  ONEHOT:
    NUM_CLASSES: 10
SOLVER:
  OPTIMIZER:
    BASE_LR: 0.1
    WEIGHT_DECAY_NORM: 0.0001
  SCHEDULER:
    TOTAL_EPOCHS: 164
    POLICY: "STEP"
    STEPS: (81, 122)
    WARM_UP_EPOCHS: 0
DATA:
  PIXEL_MEAN: (0.4914, 0.4822, 0.4465)
  PIXEL_STD: (0.2023, 0.1994, 0.2010)
  DATASET_TYPE: "cifar_dataset"
  SAMPLER:
    ASPECT_RATIO_GROUPING: []
TRAIN:
  DATASETS: ("cifar10",)
  BATCH_SIZE: 128
  TRANSFORMS: ("resize", "random_crop", "random_horizontal_flip", "to_tensor", "normalize")
  RESIZE:
    SCALES: (32,)
    MAX_SIZE: -1
    SCALES_SAMPLING: "choice"
  RANDOM_CROP:
    CROP_SCALES: ((32, 32),)
    BORDER: 128
TEST:
  DATASETS: ('cifar10',)
  IMS_PER_GPU: 16
  RESIZE:
    SCALE: 32
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (32, 32)
```

若想修改网络的模型结构，可根据上一步定义的参数，在yaml文件或启动的命令中指定。
