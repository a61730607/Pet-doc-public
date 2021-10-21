# 迭代优化

迭代优化是训练深度学习模型的核心内容，Pet对深度学习模型的优化迭代操作设计了一套标准的实现，将深度学习模型的训练与优化操作归纳为优化器和学习率调度器两项组合操作，并且使用了混合精度训练与PyTorch提供的分布式数据并行方法提高模型训练的效率。在Pet的代码实现中，优化器和学习率调度器具体对应`Optimizer`和`Scheduler`两个基本Python操作类，两个Python类会在整个训练的过程中一直被用于指导模型的优化。可以使用配置系统中`SOLVER`模块的设置来指导构建优化器和学习率调度器，对模型训练过程中的优化算法、学习率变化以及参数差异化优化策略进行系统设置。

默认配置文件目录：`pet/lib/config/solver.py`

## 优化器

当您在完成网络模型的构建之后，优化器可以帮助您对网络结构中不同种类的参数的学习率、权重衰减因子以及学习率倍率进行差异化设置，同时还提供一些主流的优化算法，您可以根据您不同的训练需求来配置优化器，优化器的完整代码请参考```pet/lib/utils/optimizer.py```。

优化器的实现规则与流程如下：

- 优化器对构建好的模型进行参数解析，根据需要对模型参数进行归类，不同类型的参数和模块会被分配以不同的权重衰减和学习率倍率；
- 将归类和差异化配置之后的模型参数送入torch提供的优化算法，完成优化器的配置。

Pet中优化器支持的功能：

- 支持的优化器：SGD、AdamW、Adam、RMSprop。

- 为偏置、标准化算子、嵌入层与其他网络层设置不同的```weight_decay``` 。

- 为backbone、相对位置偏执分别设置学习率缩放因子。
- 梯度裁剪（仅支持SGD与AdamW优化器）。

当您需要对网络模型中的其他参数进行差异化优化设置，或者您需要使用新的优化算法时，您需要在优化器内遵循以上的代码实现标准，将您的改进加入Pet。

### 初始化

```python
class Optimizer(object):
    def __init__(self, model: nn.Module, optimizer: CfgNode) -> None:
        self.model = model
        self.optimizer_type = optimizer.TYPE

        # lr
        self.base_lr = optimizer.BASE_LR
        self.bias_lr_factor = optimizer.BIAS_LR_FACTOR
        self.backbone_lr_factor = optimizer.BACKBONE_LR_FACTOR
        # weight decay
        self.weight_decay = optimizer.WEIGHT_DECAY
        self.weight_decay_bias = optimizer.WEIGHT_DECAY_BIAS
        self.weight_decay_norm = optimizer.WEIGHT_DECAY_NORM
        self.weight_decay_embed = optimizer.WEIGHT_DECAY_EMBED
        # momentum
        self.momentum = optimizer.MOMENTUM
        # clip gradients
        self.clip_gradients = optimizer.CLIP_GRADIENTS
        if self.clip_gradients.ENABLED:
            assert self.optimizer_type in ("SGD", "ADAMW"), "only SGD and ADAMW support clip gradients."

        self.norm_module_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
```

```Optimizer```类的构造函数主要用于读取配置文件的对应字段，并声明标注化算子以支持优化器对这些算子设置不同的```weight_decay```。

### build

在```build```方法中，`Optimizer`类会为`torch.optim.Optimizer`的优化器增加Pet支持的功能，如梯度裁剪、设置不同学习率和`weight_decay`等。

```python
def build(self) -> torch.optim.Optimizer:
    """
    Returns:
        Optimizer
    """
    def _maybe_add_full_model_gradient_clipping(_optim, clip_grad_cfg):
        # ...

    def _maybe_add_gradient_clipping(_optim, clip_grad_cfg):
        # ...

    params = self.get_params()
    if self.optimizer_type == "SGD":    # for supporting full model clip
        optimizer = _maybe_add_full_model_gradient_clipping(torch.optim.SGD, self.clip_gradients)(
            params, self.base_lr, momentum=self.momentum
        )
    elif self.optimizer_type == "ADAMW":    # for supporting full model grad clip
        optimizer = _maybe_add_full_model_gradient_clipping(torch.optim.AdamW, self.clip_gradients)(
            params, self.base_lr
        )
    elif self.optimizer_type == "ADAM":
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )
    elif self.optimizer_type == 'RMSPROP':
        optimizer = torch.optim.RMSprop(
            self.get_params(),
            momentum=self.momentum,
        )
    else:
        raise NotImplementedError(f"no optimizer type {self.optimizer_type}")

    if not self.clip_gradients.CLIP_TYPE == "full_model" and self.optimizer_type in ("SGD", "ADAMW"):
        # for supporting value / norm grad clip
        optimizer = _maybe_add_gradient_clipping(optimizer, self.clip_gradients)

    return optimizer
```

## 调度器

调度器在训练的过程负责控制学习率。您在配置系统中`SOLVER`部分设定的学习率变化策略在训练过程的每一次迭代中计算新的基础学习率，并对模型中的不同参数调整其差异化学习率，调度器的完整代码请参考`pet/lib/utils/lr_scheduler.py`的`LearningRateScheduler`类。

`Scheduler`继承`torch.optim.lr_scheduler._LRScheduler`类。该类定义了学习率相关操作，包括多种学习率下降，warm_up等。在训练过程中，调度器会根据训练进度，动态地调整学习率，所以通过Hook机制可以方便地在特定阶段调用相应功能。

在Pet中，调度器目前支持以下功能：

- 学习率下降策略（step/cosine/step_cosine/poly）。
- warmup策略（constant/linear）

当您需要使用其他的学习率变化策略时，您需要在学习率调度器内遵循以上的代码实现标准，将您的改进加入Pet。

### 初始化

调度器在Pet中对应`LearningRateScheduler`这一具体的Python类，接收`optimizer`和`solver`（配置文件）作为输入，`LearningRateScheduler`的主要成员函数包括`get_lr`、`update_learning_rate`以及`step`。在了解`LearningRateScheduler`的功能函数之前，我们首先对`Optimizer`类进行初始化：

```python
class LearningRateScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, solver: CfgNode, start_iter: int = 0, iter_per_epoch: int = -1, last_epoch: int = -1) -> None:
        schedule = solver.SCHEDULER
        self.iteration = start_iter

        if schedule.TOTAL_EPOCHS is None:
            self.max_iter = schedule.TOTAL_ITERS
            self.milestones = schedule.STEPS
        else:
            # Convert the epoch style parameters to corresponding iteration.
            self.max_iter = schedule.TOTAL_EPOCHS * iter_per_epoch
            self.milestones = tuple(epoch * iter_per_epoch for epoch in schedule.STEPS)  # only useful for step policy

        if schedule.WARM_UP_EPOCHS is None:
            self.warmup_iters = schedule.WARM_UP_ITERS
        else:
            # Convert the epoch style parameters to corresponding iteration.
            self.warmup_iters = schedule.WARM_UP_EPOCHS * iter_per_epoch

        assert list(self.milestones) == sorted(self.milestones)

        self.base_lr = solver.OPTIMIZER.BASE_LR
        self.policy = schedule.POLICY
        self.gamma = schedule.GAMMA
        self.warmup_factor = schedule.WARM_UP_FACTOR
        self.warmup_method = schedule.WARM_UP_METHOD
        self.log_lr_change_threshold = schedule.LOG_LR_CHANGE_THRESHOLD
        self.lr_pow = schedule.LR_POW
        self.lr_factor = 0.0
        self.info = dict(best_acc=0.0, best_epoch=1, cur_acc=0.0, cur_epoch=1)

        assert self.policy in ('STEP', "COSINE", 'STEP_COSINE', 'POLY')
        assert self.warmup_method in ('CONSTANT', 'LINEAR')

        super(LearningRateScheduler, self).__init__(optimizer, last_epoch)
```

根据训练目标，调度器的设置有两种模式：iterations或epochs。在Pet中，训练目标默认是设定总的iterations。在调度器的设置中，会根据配置文件中的iterations或epochs（根据每个epoch的iterations，转化为iterations后）设置调度器学习率下降和warmup对应的iterations。

同时在初始化的过程中，Pet会检查输入的策略字段是否是Pet支持的策略，这是Pet的一种异常预警机制，可以在您需要扩展新的代码时，提醒您在训练脚本中可能遗忘或错误地定义优化器，这种预警机制被广泛运用在目前的深度学习平台和算法工程中。

### step

在对学习率调度器进行初始化之后，需要构建学习率调度器在迭代过程中的成员函数`step`。训练时通过调用`step`函数可以在每一次迭代中，根据学习率优化策略以及当前迭代数计算学习率，并将当前学习率分配给网络模型中不同的参数。

step方法会调用成员函数 get_lr 来获取当前iteration的学习率。在Pet中，step方法继承了`torch.optim.lr_scheduler._LRScheduler` 中的step方法（更详细的代码即过程请参考pytorch文档或源码），对get_lr进行了重写。

#### get_lr

```python
def get_lr(self) -> List[float]:
    """Update learning rate
    """
    warmup_factor = self.get_warmup_factor(
        self.warmup_method, self.iteration, self.warmup_iters, self.warmup_factor
    )
    if self.policy == "STEP":
        lr_factor = self.get_step_factor(warmup_factor)
    elif self.policy == "COSINE":
        lr_factor = self.get_cosine_factor(warmup_factor)
    elif self.policy == 'STEP_COSINE':
        if self.iteration < self.milestones[-1]:
            lr_factor = self.get_step_factor(warmup_factor)
        else:
            lr_factor = self.get_cosine_lrs(warmup_factor)
    elif self.policy == 'POLY':
        lr_factor = self.get_poly_factor(warmup_factor)
    else:
        raise KeyError(f'Unknown SOLVER.SCHEDULER.POLICY: {self.policy}')

    ratio = LearningRateScheduler._get_lr_change_ratio(lr_factor, self.lr_factor)
    if self.lr_factor != lr_factor and ratio > self.log_lr_change_threshold:
        if lr_factor * self.base_lr > 1e-7 and self.iteration > 1:
            logging_rank('Changing learning rate {:.7f} -> {:.7f}'.format(
                self.lr_factor * self.base_lr, lr_factor * self.base_lr)
            )
    self.lr_factor = lr_factor

    self.iteration += 1

    return [lr_factor * base_lr for base_lr in self.base_lrs]
```

在该方法中，分别根据warm_up策略和学习率下降的策略调用个对应方法，得到学习率系数（基于配置文件的base_lr）。同时针对学习率的修改，在log文件中进行输出。该方法返回一个学习率列表。

##### 学习率预热策略(Learning rate warming up)

Pet将学习率预热策略收纳于学习率调度器中，提供了**连续**（CONSTANT）和**线性**（LINEAR）两种学习率预热策略。

在当前深度学习模型的训练过程中，批量优化技术已经成为一种通用的训练方法，但是小批量的数据不足以代表整个用于训练的数据集的统计分布，当学习率设置不合理时，模型的优化方向可能并不是全局最优，这可能导致模型在迭代优化过程中出现局部最优或者是不收敛的情况。学习率预热策略在训练的开始阶段将学习率保持在一个比较小的水平，并在最大预热迭代次数之内使学习率缓慢增长，保证模型在优化的最开始不会偏向错误的方向。

##### 学习率下降策略

Pet为深度卷积神经网络模型的训练和优化提供了**阶段下降**、**余弦下降**、**阶段+余弦下降**、**复数下降**三种学习率下降策略，他们在配置系统的`SOLVER`模块中对应的字段分别是`STEP`、`COSINE`、`STEP_COSINE`和`POLY`。

## 训练过程

### 初始化

在 `tools/cnn/train_net.py` 中，通过一下代码对优化器和调度器进行实例化和初始化：

```python
optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
optimizer = checkpointer.load_optimizer(optimizer)

scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
scheduler = checkpointer.load_scheduler(scheduler)
```

并且将其加入训练阶段Hook中：

```python
all_hooks = build_train_hooks(
        cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False, precise_bn_args=precise_bn_args
    )
```

### 更新

在 ```build_train_hooks```中，通过`LRScheduler`类可以动态地在训练过程对`Scheduler`进行调整。

具体地，在`LRScheduler`中，定义了在每一个训练step后的操作，通过调用调度器的`step`方法来更新学习率：

```python
def after_step(self, storage, *args, **kwargs):
    lr = self.optimizer.param_groups[self._best_param_group_id]["lr"]
    storage.put_scalar("lr", lr, smoothing_hint=False)
    self.scheduler.step()
```