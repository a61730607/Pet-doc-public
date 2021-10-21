# Hook

钩子编程 (hooking) ，是计算机程序设计术语，指通过拦截软件模块间的函数调用、消息传递、事件传递来修改或扩展操作系统、应用程序或其他软件组件的程序执行流程。 其中，处理被拦截的函数调用、事件、消息的代码，被称为钩子 (hook)。

在Pet中，通过Hook机制，可以方便地在模型训练、测试的对应阶段，统一的进行操作。

所有的Hook类定义在 `pet/lib/utils/logger.py` 中，包括三个基础类：`AverageMeter`，`TrainHook`，`TestHook` 和继承`TrainHook`的4个类。

## TrainHook

TrainHook类仅定义了4个空的函数，对应在训练前后与每个step前后调用。

```python
class TrainHook(object):
    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs):
        pass
```

定义了4个继承TrainHook的类，来参与训练阶段设置。

- PeriodicWriter：周期性写事件
- IterationTimer：记录训练相关iter及时间
- LRScheduler：调用Scheduler
- PreciseBN：冻结BN层

## TestHook

TestHook用于跟踪重要的Test数据，记录测试相关时间与log。

## AverageMeter

计算并存储均值以及总值。