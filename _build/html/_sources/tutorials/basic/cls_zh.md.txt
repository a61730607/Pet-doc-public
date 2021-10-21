# 在ImageNet数据集上训练分类模型

### 一、介绍

本部分以Resnet50b在ImageNet数据集上的训练和测试为例，讲解检测任务的实验流程。通过本教程，您将学会如何结合Pet提供的组件进行分类任务模型
的训练。在此期间，我们只进行组件的调用，组件的实现细节您可以参阅组件部分。

### 二、快速开始

如果您已经有分类任务的训练经验，您可以直接使用[$Pet/tools/train_net_all.py](https://github.com/BUPT-PRIV/Pet/blob/master/tools/train_net_all.py)脚本开始训练。

### 三、用法示例

```shell
python tools/train_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml --gpu_id=0,1,2,3
```

在开始相关训练之前，我们的Pet需要指定一个`YAML`文件，该文件里包含了所有训练时使用到的可以调节的参数，包括学习率、GPU数量、网络结构、全数据迭代次数等等。此次教程以`$Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml`为例，以训练脚本为主，讲解训练过程中需要的关键配置以及训练步骤，详细参数见[此处](https://github.com/BUPT-PRIV/Pet/blob/master/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml)。

### 四、数据准备

在[IMAGENET](https://image-net.org/challenges/LSVRC/2017/)官网下载数据集`ILSVRC2017`，提倡将数据集放在项目外，并通过软链接的方式链接到`Pet/data/`目录下。

> 补充软链接的用法：
>
>```
>  ln -s 原始路径 目的路径
>  ```
>  注意：原始路径必须写绝对路径。



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

### 五、数据加载及预处理

yaml配置文件的数据集部分信息(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
DATA:
  PIXEL_MEAN: (0.485, 0.456, 0.406)
  PIXEL_STD: (0.229, 0.224, 0.225)
  FORMAT:
    IMAGE: "rgb"
  DATASET_TYPE: "image_folder_dataset"
  SAMPLER:
    ASPECT_RATIO_GROUPING: []
TRAIN:
  DATASETS: ("imagenet1k_2017_train",)
  BATCH_SIZE: 256
  TRANSFORMS: ("random_resized_crop", "random_horizontal_flip", "to_tensor", "normalize")
  RANDOM_RESIZED_CROP:
    SIZE: (224, 224)
TEST:
  DATASETS: ('imagenet1k_2017_val',)
  IMS_PER_GPU: 16
  RESIZE:
    SCALE: 256
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (224, 224)
```

根据配置文件构建训练数据集和加载器,核心调用`/pet/vision/datasets/dataset.py`下的`build_dataset`和`make_{train/test}_data_loader`

```python
from pet.vision.datasets.dataset import build_dataset, make_train_data_loader
from pet.vision.datasets.dataset import build_dataset, make_test_data_loader

# 构建训练数据集和加载器
dataset = build_dataset(cfg, is_train=True)
start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter)
...
# 构建测试数据集和加载器
dataset = build_dataset(cfg, is_train=False)
test_loader = make_test_data_loader(cfg, dataset)
```

构建数据集的核心是通过解析配置文件相关参数，构建`cfg.DATA.DATASET_TYPE`对应的数据集类(相关代码在`/pet/lib/data/datasets/`)，构建数据集的核心代码如下：

```python
def build_dataset(cfg, is_train=True):
    dataset_list = cfg.TRAIN.DATASETS if is_train else cfg.TEST.DATASETS
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

    trans = build_transforms(cfg, is_train=is_train)

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



### 六、构建实验配置文件

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



### 七、从配置文件中构建模型

配置文件的模型搭建部分信息(更多默认配置在`/pet/lib/config/model`)：

```yaml
MODEL:
  BACKBONE: "resnet"
  GLOBAL_HEAD:
    CLS:
      ONEHOT_ON: True
  RESNET:
    CLS_ON: True
    LAYERS: (3, 4, 6, 3)
    STRIDE_3X3: True
    NORM: "BN"
    ZERO_INIT_RESIDUAL: False
    FREEZE_AT: 0
  ONEHOT:
    NUM_CLASSES: 1000
```

根据yaml配置文件，通过GeneralizedCNN类实例化对应模型，在forward函数中按顺序控制数据传输。具体代码在`/pet/vision/modeling/model_builder.py`中：

```python
class GeneralizedCNN(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(GeneralizedCNN, self).__init__()

        self.cfg = cfg

        # Backbone
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = Backbone(cfg)
        dim_in = self.backbone.dim_out
        spatial_in = self.backbone.spatial_out

        # Neck
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



### 八、构建优化器及调度器

yaml配置文件的优化器及调度器部分信息(更多默认配置在`/pet/lib/config/solver.py`)：

```yaml
SOLVER:
  OPTIMIZER:
    BASE_LR: 0.1
    WEIGHT_DECAY_NORM: 0.0001
  SCHEDULER:
    TOTAL_EPOCHS: 120
    STEPS: (30, 60, 90)
    WARM_UP_EPOCHS: 0
```

和其他组件一样，通过解析配置文件相关参数，传给`Optimizer`类(`/pet/lib/utils/analyser.py`)和`LearningRateScheduler`类(`/pet/lib/utils/lr_scheduler.py`),从而构建优化器及调度器，仅在训练阶段使用：

```python
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

 

### 九、训练和测试

在通过实验配置定义各组件及训练流相关操作后，我们就可以开始实验的训练与测试了。在运行相关代码前，还需要先准备数据集和相关权重文件，大致步骤如下：

(1) 数据集下载

详见前面**<u>数据准备</u>**部分。

(2) 预训练模型权重/测试模型权重文件可以下载

可从 Model Zoo 中下载你所需要的权重文件到 "/ckpts/" 相应目录下

(3) 运行训练与测试代码

训练与测试的yaml配置(更多默认配置在`/pet/lib/config/data.py`)：

```yaml
TRAIN:
  DATASETS: ("imagenet1k_2017_train",)
  BATCH_SIZE: 256
  TRANSFORMS: ("random_resized_crop", "random_horizontal_flip", "to_tensor", "normalize")
  RANDOM_RESIZED_CROP:
    SIZE: (224, 224)
TEST:
  DATASETS: ('imagenet1k_2017_val',)
  IMS_PER_GPU: 16
  RESIZE:
    SCALE: 256
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (224, 224)
```

- 训练

```shell
# 8 GPUs
python tools/train_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml

# Specify GPUs
python tools/train_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml --gpu_id=0,1,2,3
```

- 测试

```shell
# Specify GPUs
python tools/test_net_all.py --cfg=cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml --gpu_id=0
```
