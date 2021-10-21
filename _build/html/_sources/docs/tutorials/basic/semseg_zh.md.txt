# 在ADE20K数据集上训练resnet50+PSP模型

### 一、介绍

本教程将介绍使用Pet训练以及测试resnet50+PSP模型进行语义分割的主要步骤，在此我们会指导您如何通过组合Pet的提供的组件来训练resnet50+PSP模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。

在阅读本教程的之前我们强烈建议您阅读原始论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)[1]、[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)[2]以了解更多关于resnet50+PSP模型的算法原理。

### 二、快速开始

如果您已经有语义分割任务的训练经验，您可以直接在Pet中运行`$Pet/tools/vision/train_net.py`脚本开始训练您的resnet50+PSP模型。

### 三、用法示例

- **训练**

直接在Pet中运行以下代码开始训练您的模型

```
cd $Pet

Python tools/vision/test_net.py --cfg cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml

# 指定GPU参数进行训练
cd $Pet

Python tools/vision/test_net.py --cfg cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml --gpu_id 0,1,2,3
```

- **测试**

在Pet中运行以下代码开始测试您的模型

```
#默认为8GPU进行训练
cd $Pet

python tools/vision/train_net.py --cfg cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x/PSPNet_R-50c_1x.yaml

# 指定GPU参数进行训练
cd $Pet

python tools/vision/train_net.py --cfg cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml --gpu_id 0,1,2,3
```



### 四、构建实验配置文件

在进行任何与模型训练和测试有关的操作之前，需要指定一个yaml文件，明确在训练时对数据集、模型结构、优化策略以及训练时可以调节的重要参数的设置，包括学习率，GPU数量、网络结构、数据迭代次数等。本教程以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`模型为例讲解训练过程中所需要的关键配置，这套配置将指导resnet50+PSPnet模型以及测试的全部步骤与细节，全部参数设置详见`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`

Pet以yaml文件格式定义并存储本次实验配置信息，并根据任务类型和所用数据集等信息将其保存在cfgs目录下相应位置。以`cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x/PSPNet_R-50c_1x.yaml`为例，其包含的大致配置信息如下（所有默认基础配置可在pet/lib/config/目录下查询）

```python
MISC:# 基础配置
    ...
MODEL:# 模型配置
    ...
SOLVER:# 优化器及调度器配置
    ...
DATA:# 数据相关配置
    ...
TRAIN:# 训练配置
    ...
TEST:# 测试配置
    ...
```



### 五、数据集准备与介绍

确保ADE20K数据集已经存放在您的硬盘中并整理好文件结构。

[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)[3]是一个场景解析数据集，该数据集包含27611幅图片，这些图像用开放字典标签集密集注释，并附有150个类别对象的标注信息。对于2017 Places Challenge 2，选择了覆盖89％所有像素的100个thing和50个stuff类别。

数据集组成

| Database        | Number       | illustration                                                 |
| --------------- | ------------ | ------------------------------------------------------------ |
| Training Set    | 25547 images | All images are fully annotated with objects and, many of the images have parts too. |
| Validation Set  | 2000 images  | Fully annotated with objects and parts                       |
| Test Set        | \            | Images to be released later.                                 |
| Consistency set | 64 images    | 64 images and annotations used for checking the annotation consistency |



### 六、数据加载

在您的硬盘准备好ADE20K数据集，文件夹结构如下

```
ADE20K
	|--images
		|--testing
		|--training
		|--validation
		...
	|--annotations
		|--training
		|--validation
		|--visualization
		...
```

预训练模型权重/测试模型权重下载

```
从MOdel Zoo中下载所需要的权重文件到"/ckpts/"相应目录下
```

yaml文件中关于数据集部分的配置如下

```python
DATA:
  PIXEL_MEAN: (0.485, 0.456, 0.406)# 像素平均值（BGR顺序）作为元组
  PIXEL_STD: (0.229, 0.224, 0.225)# 像素标准差（BGR顺序）作为元组
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



### 七、从配置文件中进行模型构建

以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`为例，其包含了基础配置、模型配置等基本配置信息。

关于使用模型的详细配置解释参见`$Pet/lib/config/model/backbone.py`

resnet50+PSP模型构建的配置信息如下

```python
MISC:# 基础配置
  CKPT: "ckpts/vision/ADE20K/pspnet/PSPNet_R-50c_1x"#权重文件路径
MODEL:# 模型配置
  BACKBONE: "resnet"# 骨干网络配置
  NECK: ""
  GLOBAL_HEAD:# 任务配置，本实验为语义分割，对应SEMSEG
    SEMSEG:
      AUXSEG_ON: True
      PSPNET_ON: True
  RESNET:# 骨干网络resnet的结构设计
    LAYERS: (3, 4, 6, 3)# 每一模块的层数，此处的参数设置为resnet50
    STRIDE_3X3: True
    USE_3x3x3HEAD: True
    NORM: "SyncBN"
    STRIDE: 8
    FREEZE_AT: 0
  AUXSEG:# GLOBAL_HEAD的AUXSEGMoudle的构建参数
    CONV_DIM: 1024
    NORM: "SyncBN"
    NUM_CLASSES: 150
    IGNORE_LABEL: 255
    LABEL_DOWN_SAMPLE_RATE: 1
    LOSS_WEIGHT: 0.4
  PSPNET:# GLOBAL_HEAD的PSPMoudle的构建参数
    NORM: "SyncBN"
    NUM_CLASSES: 150
    IGNORE_LABEL: 255
    LABEL_DOWN_SAMPLE_RATE: 1
    LOSS_WEIGHT: 1.0
```

根据yaml配置文件，通过GeneralizedCNN类实例化对应模型，在forward函数中按顺序控制数据传输。具体代码在`/pet/vision/modeling/model_builder.py`中：

```python
from pet.vision.modeling.model_builder import GeneralizedCNN

class GeneralizedCNN(nn.Module):
    
        """ 视觉模型构建+前向函数定义    """
    def __init__(self, cfg: CfgNode) -> None:
        super(GeneralizedCNN, self).__init__()

        self.cfg = cfg
        # 构建backbone部分：ResNet
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = Backbone(cfg) 
        ...
        # Neck为""，无需构建此部分
        if cfg.MODEL.NECK: 
            Neck = registry.NECKS[cfg.MODEL.NECK]
            ...
        ...
        # 构建semseg的检测头
        if cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON:# cfg.infer_cfg()调用
            self.global_semseg = GlobalSemSeg(cfg, dim_in, spatial_in)
        ...
    ...
```



### 八、优化器与调度器的构建

以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`为例，在优化器中对基本学习率进行了设定，在调度器中设定了最大迭代次数、SGD迭代次数、调度器类型。

关于优化器与调度器的构建详细配置解释参见`$Pet/lib/config/solver.py`

yaml文件中关于优化器与调度器的配置信息如下：

```python
SOLVER:
  OPTIMIZER:# 优化器
    BASE_LR: 0.01# 基本学习率
  SCHEDULER:
    TOTAL_ITERS: 151575  # 20210 * 120 / 16 = 151575，最大迭代次数
    WARM_UP_ITERS: 0# SGD迭代次数，预热到SOLVER.OPTIMIZER.BASE_LR
    POLICY: "POLY"# 调度器类型，这里使用的是POLY，其他还有"STEP", "COSINE", ...
```

通过解析配置文件相关参数，传给`Optimizer`类(`/pet/lib/utils/analyser.py`)和`LearningRateScheduler`类(`/pet/lib/utils/lr_scheduler.py`),从而构建优化器及调度器，仅在训练阶段使用：

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



### 九、模型训练参数配置

以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`为例，模型主要的训练流程有指定权重文件路径、指定训练集、指定训练过程中需要用到的数据预处理参数、指定图像增强参数、指定随机裁剪参数等，在该yaml文件中对这部分参数进行了指定。

关于训练部分的详细配置解释参见`$Pet/lib/config/data.py`

```python
TRAIN:# 训练参数设定
  WEIGHTS: "ckpts/vision/ImageNet/3rdparty/resnet/resnet50c_mmcv/resnet50_v1c-2cccc1ad-convert.pth"# 指定权重文件路径
  DATASETS: ("ade2017_sceneparsing_train",)#指定训练集
  SIZE_DIVISIBILITY: 8# 指定每一个整理批次的分割数
  TRANSFORMS: ("color_jitter", "resize", "random_crop", "random_horizontal_flip", "to_tensor", "normalize")# 训练过程中需要用到的数据预处理参数
  COLOR_JITTER:# 图像增强参数
    BRIGHTNESS: 0.4
    CONTRAST: 0.4
    SATURATION: 0.4
    HUE: 0.1
  RANDOM_CROP:# 随机裁剪参数
    CROP_SCALES: ((512, 512),)# 随机裁剪的比例，如果img_size<scale，则使用PAD_像素填充间隙。(H, W)必须能被SIZE_DIVISIBILITY整除，默认为((640, 640),)
    CAT_MAX_THS: 0.75# 裁剪区域选择的CAT_MAX_THS
    IGNORE_LABEL: 255# 忽略cat max像素计算的标签
  RESIZE:
    SCALES_SAMPLING: "scale_factor"# 训练期间最小最小尺寸的采样类型，这里使用的是"scale_factor"，其余还有"choice", "range", .
    SCALE_FACTOR: (0.5, 0.75, 1, 1.25, 1.5, 1.75, 2)
```

关于模型训练的主要步骤包括创建模型、创建检查点、加载预训练权重或随机初始化、创建优化器、创建训练集与加载器、构建调度器、模型分布式等。以下代码列出了部分训练步骤，详细参见`$pet/tools/vision/train_net.py`。

```python
# Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.ANALYSER.ENABLED:
        model = GeneralizedCNN(cfg)
        model.eval()
        analyser = RCNNAnalyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.MAX_SIZE, mode="flops")
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.MAX_SIZE, mode="activations")
        del model
    synchronize()

    # Create model
    model = GeneralizedCNN(cfg)
    logging_rank(model)
    logging_rank(
        "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | Activations: {:.4f}M / Conv_Activations: {:.4f}M"
        .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
    )

    # Create checkpointer
    checkpointer = CheckPointer(cfg.MISC.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME)

    # Load pre-trained weights or random initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MISC.CONV1_RGB2BGR)
    model.to(torch.device(cfg.MISC.DEVICE))
    if cfg.MISC.DEVICE == "cuda" and cfg.MISC.CUDNN:
        cudnn.benchmark = True
        cudnn.enabled = True

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank("The mismatch keys: {}".format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))))

    # Create training dataset and loader
    dataset = build_dataset(cfg, is_train=True)
    start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
    train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter)
    max_iter = len(train_loader)
    if cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS is None:
        iter_per_epoch = -1
    else:
        iter_per_epoch = max_iter // cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS

    # Some methods need to know present iter
    cfg.defrost()
    cfg.SOLVER.START_ITER = start_iter
    cfg.SOLVER.SCHEDULER.TOTAL_ITERS = max_iter
    if cfg.SOLVER.SNAPSHOT_EPOCHS is not None:
        cfg.SOLVER.SNAPSHOT_ITER = cfg.SOLVER.SNAPSHOT_EPOCHS * iter_per_epoch
    cfg.freeze()

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Precise BN
    precise_bn_args = [
        make_train_data_loader(cfg, dataset, start_iter=start_iter), model,
        torch.device(cfg.MISC.DEVICE)
    ] if cfg.TEST.PRECISE_BN.ENABLED else None

    # Model Distributed
    distributed = get_world_size() > 1
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Build hooks
    if cfg.SOLVER.SCHEDULER.WARM_UP_EPOCHS is None:
        warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_ITERS
    else:
        warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_EPOCHS * iter_per_epoch
    all_hooks = build_train_hooks(
        cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False, precise_bn_args=precise_bn_args
    )

    # Train
    train(cfg, model, train_loader, optimizer, scheduler, checkpointer, all_hooks)
```



### 十、模型测试

以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`为例，模型测试过程中需要指定测试集，指定图像大小调整的参数等，这部分在yaml文件中有详细的配置。

关于测试部分的详细配置解释参见`$Pet/lib/config/data.py`

```python
TEST:# 测试参数设定
  DATASETS: ("ade2017_sceneparsing_val",)# 指定测试集
  SIZE_DIVISIBILITY: 8# 指定每一个整理批次的分割数
  RESIZE:
    SCALE: 512# 测试期间图像大小调整的参数，是图像最短边的像素大小
```

关于模型测试的主要步骤包括创建模型、加载模型、创建测试数据集与加载器、构建测试引擎等。详细参见`$pet/tools/vision/test_net.py`。

```python
# Load model
    test_weights = get_weights(cfg.MISC.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, test_weights)
    model.eval()
    model.to(torch.device(cfg.MISC.DEVICE))

    # Create testing dataset and loader
    dataset = build_dataset(cfg, is_train=False)
    test_loader = make_test_data_loader(cfg, dataset)

    # Build hooks
    all_hooks = build_test_hooks(args.cfg_file.split("/")[-1], log_period=10, num_warmup=0)

    # Build test engine
    test_engine = TestEngine(cfg, model, dataset)

    # Test
    test(cfg, test_engine, test_loader, dataset, all_hooks)
```



### 十一、可视化与指标

以`$Pet/cfgs/vision/ADE20K/pspnet/PSPNet_R-50c_1x.yaml`为例，模型的评估需要存储测试记录，设定评估参数，这部分在yaml文件中有详细的配置。

关于评估部分的详细配置解释参见`$Pet/lib/config/config.py`

```python
EVAL:# 验证
  RECORD: [{"time": "20210512", "recorder": "user", "version": "0.6c",
            "semseg": "mIoU/MeanACC/PixelACC/MeanF1Score:41.91/52.37/80.17/55.83",
            "mark": ""},
           {"time": "20210516", "recorder": "user", "version": "0.7a",
            "semseg": "mIoU/MeanACC/PixelACC/MeanF1Score:42.26/52.61/80.05/56.08",
            "mark": ""},
           {"time": "20210617", "recorder": "user", "version": "0.7a",
            "semseg": "mIoU/MeanACC/PixelACC/MeanF1Score:41.91/52.06/79.90/55.75",
            "mark": ""}]# 测试记录存储，"time":测试时间；"recorder":测试者；"version":所用版本；"semseg": "mIoU/MeanACC/PixelACC/MeanF1Score:41.91/52.37/80.17/55.83":评估参数
```

### 参考文献

[1] He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[2] Zhao, Hengshuang, et al. "Pyramid scene parsing network." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

[3] Zhou, Bolei, et al. "Scene parsing through ade20k dataset." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.
