# 在ImageNet数据集上训练ResNet50分类模型



## 一、介绍

本教程将介绍使用Pet训练以及测试Resnet50模型进行图像分类的主要步骤，在此我们会指导您如何通过组合Pet的提供组件来训练Mask R-CNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。

在阅读本教程的之前我们强烈建议您阅读原始论文[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)[1]、[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)[2]和[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)[3]以了解更多关于ResNet的算法原理。 

数据集使用[IMAGENET](https://image-net.org/challenges/LSVRC/2017/)。



## 二、快速开始 

如果您已经有分类任务的训练经验，您可以直接使用[Pet/tools/train_net_all.py](https://github.com/BUPT-PRIV/Pet/blob/master/tools/train_net_all.py)脚本开始训练。

首先进入`Pet`，

```shell
cd Pet
```

用法示例：

```shell
# 使用8块GPU进行训练
python tools/train_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
# 使用指定的4块GPU进行训练
python tools/train_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml --gpu_id=0,1,2,3‘

# 使用8块GPU进行测试
python tools/test_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
# 使用特定的GPU进行测试
python tools/test_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml --gpu_id=0
```

## 三、构建实验配置文件

在进行任何与模型训练和测试有关的操作之前，需要先指定一个YAML文件，明确在训练时对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以`$Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml`为例，讲解训练过程中所需要的关键配置，该套配置将指导此ReNet50模型训练以及测试的全部步骤和细节，全部参数设置请见以`$Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml`。  

Pet以yaml文件格式定义并存储本次实验配置信息，并根据任务类型和所用数据集等信息将其保存在`/cfgs`目录下的相应位置。以`/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml`为例，其包含的大致配置信息如下(所有默认基础配置可在`/pet/lib/config/`目录下查询)：

```yaml
MISC: # 基础配置
  ...
MODEL: # 模型配置
  ...
SOLVER: # 优化器及调度器配置
  ...
DATA: # 数据相关配置
  ...
TRAIN: # 训练配置
  ...
TEST: # 测试配置
  ...
EVAL: # 评价配置
  ...
```



## 四、数据准备和介绍

在[官网IMAGENET](https://image-net.org/challenges/LSVRC/2017/)下载数据集`ILSVRC2017`，**强烈推荐**将数据集放在项目外，并通过**软链接**的方式链接到`Pet/data/`目录下。

确保ImageNet数据已经存入您的电脑硬盘中，三级文件结构（仅列出目录，不含文件）如下：

```
ILSVRC2017
├── Annotations
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   └── DET
│       ├── train
│       └── val
├── Annotations_refine
│   └── DET
│       ├── train
│       └── val
├── Data
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   ├── DET
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── DET_CROP
│       ├── train
│       └── val
├── ImageSets
│   ├── CLS-LOC
│   └── DET
└── log
    ├── DET
    └── VID
```

分类需要用到的部分有：

- 数据集：`ILSVRC2017/Data/CLS-LOC/`
- 标签：`ILSVRC2017/Annotations/CLS-LOC/`



## 五、数据加载

yaml配置文件的数据集部分信息(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
## $Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
DATA:
  PIXEL_MEAN: (0.485, 0.456, 0.406)
  PIXEL_STD: (0.229, 0.224, 0.225)
  # 默认配置，在pet/lib/config/data.py中
  DATASET_TYPE: "image_folder_dataset"
...
TRAIN:
  DATASETS: ("imagenet1k_2017_train",)
  # 数据增强
  TRANSFORMS: ("random_resized_crop", "random_horizontal_flip", "to_tensor", "normalize")
  RANDOM_RESIZED_CROP:
    SIZE: (224, 224)
TEST:
  DATASETS: ('imagenet1k_2017_val',)
  RESIZE:
    SCALE: 256
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (224, 224)
```



根据配置文件构建训练数据集和加载器,核心调用`/pet/vision/datasets/dataset.py`下的`build_dataset`和`make_{train/test}_data_loader`

```python
## $Pet/tools/vision/train_net.py
from pet.vision.datasets.dataset import build_dataset, make_train_data_loader
from pet.vision.datasets.dataset import build_dataset, make_test_data_loader

# 构建训练数据集和加载器
dataset = build_dataset(cfg, is_train=True)
start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter)
...

```

```python
## $Pet/tools/vision/test_net.py

# 构建测试数据集和加载器
dataset = build_dataset(cfg, is_train=False)
test_loader = make_test_data_loader(cfg, dataset)
```



构建数据集的核心是通过解析配置文件相关参数，构建`cfg.DATA.DATASET_TYPE`对应的数据集类(相关代码在`/pet/lib/data/datasets/`)，构建数据集的核心代码如下：

```python
## $Pet/pet/tasks/cls/datasets/dataset.py

def build_dataset(cfg, is_train=True):
    dataset_list = cfg.TRAIN.DATASETS if is_train else cfg.TEST.DATASETS # 要读取的数据集list，如：("imagenet1k_2017_train",)/('imagenet1k_2017_val',)
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    ds_list = []
    for dataset_name in dataset_list:
        assert contains(dataset_name), \
            'Unknown dataset name: {}'.format(dataset_name)
        assert os.path.exists(get_im_dir(dataset_name)), \
            'Im dir \'{}\' not found'.format(get_im_dir(dataset_name))
        ds_list.append(get_im_dir(dataset_name))
        logging_rank('Creating: {}'.format(dataset_name))

    trans = build_transforms(cfg, is_train=is_train)   # 根据配置信息，对数据做预处理变换

    if dataset_list[0] == 'cifar10':  # cifar10
        data_set = CIFAR10(root=get_im_dir(dataset_list[0]), train=is_train, transform=trans)
    elif dataset_list[0] == 'cifar100':  # cifar100
        data_set = CIFAR100(root=get_im_dir(dataset_list[0]), train=is_train, transform=trans)
    else:
        # ImageNet
        data_set = ImageFolderList(ds_list, transform=trans)

    return data_set
```

构建数据加载器的核心调用是`torch.utils.data.DataLoader`,根据配置参数解析，确定 `DataLoader` 的各参数，包括`batch_size`,`sampler`,`collect_fn`等，`sampler`和`collate_fn`的自定义实现分别在`/pet/lib/data/samplers`和`/pet/lib/data/collate_batch.py`。数据集加载分训练和测试两种情况，大致逻辑相似，为便于阅读，此处仅以测试集加载函数作为示例：

```python
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from pet.lib.data.collate_batch import BatchCollator
from pet.lib.data.samplers import (GroupedBatchSampler,
                                   IterationBasedBatchSampler,
                                   RepeatFactorTrainingSampler)

def make_test_data_loader(cfg, datasets):
    ims_per_gpu = cfg.TEST.IMS_PER_GPU
    test_sampler = DistributedSampler(datasets, shuffle=False)
    num_workers = cfg.DATA.LOADER_THREADS
    collator = BatchCollator(-1)
    data_loader = DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return data_loader
```





## 六、模型构建

配置文件的模型搭建部分信息(更多默认配置在`/pet/lib/config/model`)：

```yaml
## $Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
MODEL:
  BACKBONE: "resnet" # 骨干网络配置，根据值"resnet"找到相应骨干网络类"RESNET"，对照其参数信息进行配置补充
  GLOBAL_HEAD:  # 头部设计
    CLS:
      ONEHOT_ON: True  # 分类，ONEHOT_ON为True
  RESNET:   # 骨干网络RESNET的构建参数
    CLS_ON: True  # 分类
    LAYERS: (3, 4, 6, 3)
    STRIDE_3X3: True
    NORM: "BN"  # 批量归一化
    ZERO_INIT_RESIDUAL: False
    FREEZE_AT: 0
  ONEHOT: 
    NUM_CLASSES: 1000  # 分类的类别数量
```



根据yaml配置文件，通过GeneralizedCNN类实例化对应模型，在forward函数中按顺序控制数据传输。具体代码在`/pet/vision/modeling/model_builder.py`中：

```python
## $Pet/pet/vision/modeling/model_builder.py
class GeneralizedCNN(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(GeneralizedCNN, self).__init__()

        self.cfg = cfg

        # Backbone: 构建ResNet50的backbone部分
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = Backbone(cfg)
        dim_in = self.backbone.dim_out
        spatial_in = self.backbone.spatial_out

        # Neck为""，无需构建此部分
        if cfg.MODEL.NECK:
            Neck = registry.NECKS[cfg.MODEL.NECK]
            self.neck = Neck(cfg, dim_in, spatial_in)
            dim_in = self.neck.dim_out
            spatial_in = self.neck.spatial_out

        #  Global Head: Classification
        if cfg.MODEL.GLOBAL_HEAD.CLS_ON:  # 配置文件中的分类开关打开
            self.global_cls = GlobalCls(cfg, dim_in, spatial_in)
        ...
    ...
....
```



## 七、优化器

yaml配置文件的优化器及调度器部分信息(更多默认配置在`/pet/lib/config/solver.py`)：

```yaml
## $Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
SOLVER:
  OPTIMIZER:  # 优化器
    BASE_LR: 0.1  # 基本学习率
    WEIGHT_DECAY_NORM: 0.0001  # 学习衰减率
  SCHEDULER:
    TOTAL_EPOCHS: 120
    STEPS: (30, 60, 90)
    WARM_UP_EPOCHS: 0
```

和其他组件一样，通过解析配置文件相关参数，传给`Optimizer`类(`/pet/lib/utils/analyser.py`)和`LearningRateScheduler`类(`/pet/lib/utils/lr_scheduler.py`),从而构建优化器及调度器，仅在训练阶段使用：

```python
## $Pet/tools/vision/train_net.py
from pet.lib.utils.optimizer import Optimizer
from pet.lib.utils.lr_scheduler import LearningRateScheduler

# 构建优化器
optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
optimizer = checkpointer.load_optimizer(optimizer)
...
# 构建调度器
scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
scheduler = checkpointer.load_scheduler(scheduler)
```

 

## 八、训练

在通过实验配置定义各组件及训练流相关操作后，就可以开始模型的训练。

训练的yaml配置(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
## $Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml
TRAIN:
  DATASETS: ("imagenet1k_2017_train",)  # 训练集
  BATCH_SIZE: 256  # 批处理的数量
  # 数据增强
  TRANSFORMS: ("random_resized_crop", "random_horizontal_flip", "to_tensor", "normalize")
  RANDOM_RESIZED_CROP:
    SIZE: (224, 224)
```

训练实现的核心代码在 `$Pet/tools/vision/train_net.py` 的train函数中定义，详细介绍请看[Pet的训练教程](https://pet-doc-public.readthedocs.io/en/latest/usage/training_zh.html)。

## 九、测试

测试的yaml配置(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
## $Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml

TEST:
  DATASETS: ('imagenet1k_2017_val',)
  RESIZE:
    SCALE: 256
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (224, 224)
```

测试实现的核心代码在 `$Pet/tools/vision/test_net.py` 的test函数中定义，详细介绍请看[Pet的测试教程](https://pet-doc-public.readthedocs.io/en/latest/usage/evaluation_zh.html)。


## 十、可视化

yaml的部分配置(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
MISC:
  CKPT: "ckpts/vision/ImageNet/resnet/resnet50b_vision"
  CUDNN: True
```

模型的默认加载位置在上面配置中`CKPT`所指目录下的`model_latest.pth`。

核心代码位于`tools/vision/inference.py`，分类模型仅有打印结果，相关代码如下。

```python
if cfg.MODEL.GLOBAL_HEAD.CLS_ON:
        labels = im_labels[0].astype(np.int8)
        scores = im_labels[1]
        for l, s in zip(labels, scores):
            print("{}\t{}".format(dataset.classes[l], round(s, 4)))
```



## 参考文献

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.

[3] Saining Xie, Ross Girshick, Piotr Dolla ́r, Zhuowen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

