# 模型加载与保存


模型的加载与保存对网络训练十分重要，Pet定义了一个类`CheckPointer`用于相关功能的封装。在进行模型训练时，加载的模型分为之前已有的检查点模型、从其他任务中迁移过来的模型和进行随机初始化的模型，您可以设置`cfg.TRAIN.WEIGHTS`中的对应变量选择对应参数加载方式。模型加载与保存的处理内容包括了模型的网络参数还有模型训练所需要的优化器和学习率调度器。

## CheckPointer类

在您要进行模型参数加载和保存时，`CheckPointer`可以帮您将这一过程封装起来。在完成CheckPointer类的实例化之后，只需调用成员函数就能完成相应的模型加载或保存功能。完整代码请参考`pet/lib/utils/checkpointer.py`。

模型加载与保存的实现过程如下：

* 完成该类的实例化，确认模型参数的加载方式；

* 通过调用类中的成员函数进行模型参数、优化器、学习率调节器的更新；

* 完成一次迭代后，将训练出的模型参数、优化器、学习率调节器设置作为最新一次的结果进行更新。

### 初始化

`CheckPointer`的初始化用于确定模型参数的加载方式，通过对`weights_path`和`resume`的初始化来实现，进行初始化需要传入`ckpt`、`weights_path`、`auto_resume`参数。

* `ckpt`：进行参数加载的模型所在的路径。

* `weights_path`：迁移参数的模型所在路径。

* `auto_resume`：作为是否使用参数加载检查点模型的标志变量，在为True时选择最近检查点模型的权重参数进行初始化。


```Python
class CheckPointer(object):
    def __init__(self, ckpt, weights_path=None, auto_resume=True):
        self.ckpt = ckpt
        self.weights_path = weights_path
        self.auto_resume = auto_resume
        self.retrain = self.weights_path.endswith('model_latest.pth')

        self.mismatch_keys = set()
        self.resume = self.get_model_latest()
        if self.weights_path:
            self.checkpoint = self._load_file()
```

`weights_path`和`resume`两个变量在执行模型加载功能函数`load_model`时被用到，这里介绍他们与不同加载方式的对应关系：

* `resume`为True：加载检查点模型；

* `resume`为False且`weights_path`为True：加载预训练模型；

* `resume`、`weights_path`均为`False`：加载直接随机初始化后的网络参数。

#### get_model_latest

用于初始化`resume`，在初始化中被调用，通过确认类的`ckpt`成员变量下是否有目标模型`model_latest.pth`来赋予不同的值。

若在checkpoint目录下存在`model_lates.pth`且`self.auto_resume=True`，则`resume=True`，否则为`False`。

#### _load_file

调用`torch.load`加载模型参数并返回。

#### weight_mapping

转换caffe风格的VGG16模型。

#### convert_conv1_rgb2bgr

转换模型的第一个卷积核，使rgb模式训练出的模型权重支持bgr模式。

### load_model

```Python
def load_model(self, model, convert_conv1=False, use_weights_once=False):
    """
    Args:
        model (nn.Module)
        convert_conv1 (bool, optional): Defaults to False.
        use_weights_once (bool, optional): Defaults to False.

    Returns:
        nn.Module
    """
    if self.resume:
        weights_dict = self.checkpoint.pop('model')
        weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
        model_state_dict = model.state_dict()
        model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,
                                                                            use_weights_once)
        model.load_state_dict(model_state_dict)
        logging_rank('Resuming from weights: {}.'.format(self.weights_path))
    else:
        if self.weights_path:
            if not self.retrain:
                try:
                    weights_dict = self.checkpoint.pop('model')
                except:
                    weights_dict = self.checkpoint
            else:
                weights_dict = self.checkpoint.pop('model')
            weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
            weights_dict = self.weight_mapping(weights_dict)    # only for pre-training
            if convert_conv1:   # only for pre-training
                weights_dict = self.convert_conv1_rgb2bgr(weights_dict)
            model_state_dict = model.state_dict()
            model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,
                                                                                use_weights_once)
            model.load_state_dict(model_state_dict)
            logging_rank('Pre-training on weights: {}.'.format(self.weights_path))
        else:
            logging_rank('Training from scratch.')
    return model
```

模型的三种参数加载方式在`load_model`成员函数中实现，这里调用了`weight_mapping`、`convert_conv1_rgb2bgr`等成员函数和其他外部函数，下面介绍函数中三种加载方式的实现过程。

1、加载最近检查点的模型参数的执行过程：

* `checkpoint`中的网络参数赋给`weights_dict`存储；

* 使用`strip_prefix_if_present`函数去除`weights_dict`的参数前缀名；

* 将模型参数中的优化器以外的参数组成字典返回给新的参数字典`model_state_dict`；

* 使用`align_and_update_state_dicts`函数将参数映射到模型对应的参数字典`model_state_dict`中；

* 将参数字典`model_state_dict`中的参数对应加载到模型；

* 最后使用`logging_rank`函数打印模型参数的加载方式。

2、加载预训练模型参数的执行过程：

* `checkpoint`中的参数赋给`weights_dict`存储；

* 使用`strip_prefix_if_present`函数去除`weights_dict`的参数前缀名；

* 使用`weight_mapping`将预训练模型中参数的前缀替换为Pet中定义模型参数的前缀；

* 使用`convert_conv1_rgb2bgr`函数将卷积层的参数通道进行转换；

* 使用`align_and_update_state_dicts`函数将参数映射到模型的参数字典；

* 将参数字典`model_state_dict`中的参数对应加载到模型；

* 最后使用`logging_rank`函数打印模型参数的加载方式。

3、参数随机初始化的执行过程：

* 使用`logging_rank`函数打印模型参数的加载方式。


### load_optimizer

在加载最近检查点的模型参数时，将其中的优化器参数取出，更新当前优化器。

```Python
def load_optimizer(self, optimizer):
    if self.resume:
        optimizer.load_state_dict(self.checkpoint.pop('optimizer'))
        logging_rank('Loading optimizer done.')
    else:
        logging_rank('Initializing optimizer done.')
    return optimizer
```


### load_scheduler

与`load_optimizer`功能相似，用于更新当前的学习率调节器。

```Python
def load_scheduler(self, scheduler):
    if self.resume:
        scheduler.iteration = self.checkpoint['scheduler']['iteration']
        scheduler.info = self.checkpoint['scheduler']['info']
        logging_rank('Loading scheduler done.')
    else:
        logging_rank('Initializing scheduler done.')
    return scheduler
```


### save

定义了一个中间变量`save_dict`，用于保存模型本次训练的模型参数、优化器和学习率调节器设置，并将其存入成员变量`ckpt`路径下，打印模型的保存状态和位置。调用该函数需要传入`model`、`optimizer`、`scheduler`、`copy_latest`和`infix`等参数。

```Python
def save(self, model, optimizer=None, scheduler=None, copy_latest=True, infix='epoch'):
    """
    Args:
        model (nn.Module)
        optimizer (torch.optim.Optimizer, optional): Defaults to None.
        scheduler (torch.optim._LRScheduler, optional): Defaults to None.
        copy_latest (bool, optional): Defaults to True.
        infix (str, optional): Defaults to 'epoch'.
    """
    save_dict = {'model': model.state_dict()}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    torch.save(save_dict, os.path.join(self.ckpt, 'model_latest.pth'))
    logg_sstr = 'Saving checkpoint done.'
    if copy_latest and scheduler:
        shutil.copyfile(os.path.join(self.ckpt, 'model_latest.pth'),
                        os.path.join(self.ckpt, 'model_{}{}.pth'.format(infix, str(scheduler.iteration - 1))))
        logg_sstr += ' And copy "model_latest.pth" to "model_{}{}.pth".'.format(infix, str(scheduler.iteration - 1))
    logging_rank(logg_sstr)
```

| 成员变量 | 含义 |
| :-: | :-: |
| model | 当前迭代下的模型参数 |
| optimizer | 当前迭代下的优化器设置 |
| scheduler | 当前迭代下的学习率调节器 |
| copy_latest | 是否进行模型参数文件复制操作的标志变量 |
| infix | 当前迭代数 |

### save_best

保存最佳结果。该方法会编辑哦当前精度与历史最高精度大小，若超过历史最高精度则保存新的检查点。

```python
def save_best(self, model, optimizer=None, scheduler=None, remove_old=True, infix='epoch'):
    """
    Args:
        model (nn.Module)
        optimizer (torch.optim.Optimizer, optional): Defaults to None.
        scheduler (torch.optim._LRScheduler, optional): Defaults to None.
        remove_old (bool, optional): Defaults to True.
        infix (str, optional): Defaults to 'epoch'.

    Returns:
        bool
    """
    if scheduler.info['cur_acc'] < scheduler.info['best_acc']:
        return False

    old_name = 'model_{}{}-{:4.2f}.pth'.format(infix, scheduler.info['best_epoch'], scheduler.info['best_acc'])
    new_name = 'model_{}{}-{:4.2f}.pth'.format(infix, scheduler.info['cur_epoch'], scheduler.info['cur_acc'])
    if os.path.exists(os.path.join(self.ckpt, old_name)) and remove_old:
        os.remove(os.path.join(self.ckpt, old_name))
    scheduler.info['best_acc'] = scheduler.info['cur_acc']
    scheduler.info['best_epoch'] = scheduler.info['cur_epoch']

    save_dict = {'model': model.state_dict()}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    torch.save(save_dict, os.path.join(self.ckpt, new_name))
    shutil.copyfile(os.path.join(self.ckpt, new_name), os.path.join(self.ckpt, 'model_latest.pth'))
    logging_rank('Saving best checkpoint done: {}.'.format(new_name))
    return True
```