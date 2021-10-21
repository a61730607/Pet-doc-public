# Hook

钩子编程 (hooking) ，是计算机程序设计术语，指通过拦截软件模块间的函数调用、消息传递、事件传递来修改或扩展操作系统、应用程序或其他软件组件的程序执行流程。 其中，处理被拦截的函数调用、事件、消息的代码，被称为钩子 (hook)。

在Pet中，通过Hook机制，可以方便地在模型训练、测试的对应阶段，统一的进行操作。

所有的Hook类定义在 `pet/lib/utils/logger.py` 中，包括两个基础类：``TrainHook`，`TestHook` 和继承`TrainHook`的4个类。

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

- PeriodicWriter：周期性写事件。
- IterationTimer：记录训练相关iter及时间。
- LRScheduler：调用Scheduler。
- PreciseBN：使用EMA的BatchNorm实现。

#### PeriodicWriter

```python
class PeriodicWriter(TrainHook):
    """
    Write events to EventStorage periodically.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, writers, max_iter, display_iter):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects.
            max_iter (int)
            display_iter (int)
        """
        self.writers = writers
        self.max_iter = max_iter
        self.display_iter = display_iter
        for w in writers:
            assert isinstance(w, EventWriter), w

    def after_step(self, storage, *args, epoch=None, max_epoch=None, **kwargs):
        """
        Args:
            storage (EventStorage)
            epoch (int, optional): Defaults to None.
            max_epoch (int, optional): Defaults to None.
        """
        if epoch is not None:
            iter = storage.iter % self.max_iter
        else:
            iter = storage.iter

        if epoch is not None:
            storage.put_scalar("epoch", epoch, smoothing_hint=False)
        if (iter + 1) % self.display_iter == 0 or (iter == self.max_iter - 1):
            for writer in self.writers:
                writer.write(epoch=epoch, max_epoch=max_epoch)

    def after_train(self, *args, **kwargs):
        for writer in self.writers:
            writer.close()
```

`PeriodicWriter`在每个step后和训练完成后执行对应操作。在每个step后，`PeriodicWriter`会对传入的所有writers，根据iter轮次写入epoch和max_epoch。在训练结束后，对每个writer执行 `writer.close()`。

#### IterationTimer

```python
class IterationTimer(TrainHook):
    def __init__(self, max_iter, start_iter, warmup_iter, ignore_warmup_time):
        """
        Args:
            max_iter (int)
            start_iter (int)
            warmup_iter (int)
            ignore_warmup_time (bool)
        """
        self.warmup_iter = warmup_iter
        self.step_timer = Timer()
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.ignore_warmup_time = ignore_warmup_time

    def before_train(self, *args, **kwargs):
        self.start_time = time.perf_counter()
        self.total_timer = Timer()
        self.total_timer.pause()

    def after_train(self, storage, *args, **kwargs):
        iter = storage.iter
        total_time = time.perf_counter() - self.start_time
        total_time_minus_hooks = self.total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = iter + 1 - self.start_iter - self.warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logging_rank(
                "Overall training speed: {} iterations in {} ({:.6f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logging_rank(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self, *args, **kwargs):
        self.step_timer.reset()
        self.total_timer.resume()

    def after_step(self, storage, *args, **kwargs):
        # +1 because we're in after_step
        if self.ignore_warmup_time:
            # ignore warm up time cost
            if storage.iter >= self.warmup_iter:
                sec = self.step_timer.seconds()
                storage.put_scalars(time=sec)
            else:
                self.start_time = time.perf_counter()
                self.total_timer.reset()
        else:
            sec = self.step_timer.seconds()
            storage.put_scalars(time=sec)

        self.total_timer.pause()
```

`IterationTimer`负责记录训练阶段时间信息，会在训练前后和每个step前后被调用。该Hook中定义了两个计时器，`self.step_timer` 和 `self.total_timer`，分别记录每个iteration的耗时和总的训练耗时。

在训练前（后），`IterationTimer`会开始（结束）记录总的训练时间，并在结束时输出。

在每一步训练前，`self.step_timer` 会被重置，`self.total_timer` 会继续启动。在每个iteration后，会记录每个step的耗时，并暂停` self.total_timer`。

#### LRScheduler

```python
class LRScheduler(TrainHook):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer)
            scheduler (torch.optim._LRScheduler)
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self, storage, *args, **kwargs):
        lr = self.optimizer.param_groups[self._best_param_group_id]["lr"]
        storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()
```

`LRScheduler` 在每个step后被调用，记录当前的学习率，执行 `self.scheduler.step()` 以更新学习率。

#### PreciseBN

```python
class PreciseBN(TrainHook):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, precise_bn_args, period, num_iter, max_iter):
        """
        Args:
            precise_bn_args (list)
            period (int)
            num_iter (int)
            max_iter (int)
        """
        if len(get_bn_modules(precise_bn_args[1])) == 0:
            logging_rank(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self.disabled = True
            return

        self.data_loader = precise_bn_args[0]
        self.model = precise_bn_args[1]
        self.device = precise_bn_args[2]
        self.num_iter = num_iter
        self.period = period
        self.max_iter = max_iter
        self.disabled = False

        self.data_iter = None

    def after_step(self, storage, *args, epoch=None, **kwargs):
        if epoch is not None:
            next_iter = storage.iter % self.max_iter + 1
            is_final = next_iter == self.max_iter and epoch == kwargs.pop('max_epochs')
        else:
            next_iter = storage.iter + 1
            is_final = next_iter == self.max_iter
        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self.disabled:
            return

        if self.data_iter is None:
            self.data_iter = iter(self.data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    logging_rank(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self.num_iter)
                    )
                # This way we can reuse the same iterator
                yield next(self.data_iter)

        with EventStorage():
            logging_rank(
                "Running precise-BN for {} iterations...  ".format(self.num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self.model, data_loader(), self.device, self.num_iter)
```

## TestHook

TestHook用于跟踪重要的Test数据，记录测试相关时间与log。

#### 初始化

在TestHook的初始化函数中设置了相关信息，添加了各个维度的timer用于计时。

```python
class TestHook(object):
    """Track vital testing statistics."""

    def __init__(self, cfg_filename, logperiod=10, num_warmup=5):
        """
        Args:
            cfg_filename (str)
            logperiod (int, optional): Defaults to 10.
            num_warmup (int, optional): Defaults to 5.
        """
        self.cfg_filename = cfg_filename
        self.logperiod = logperiod
        self.num_warmup = num_warmup
        self.timers = OrderedDict()
        self.iter = 0

        self.default_timers = ('iter', 'data', 'infer', 'post')
        for name in self.default_timers:
            self.add_timer(name)
```

#### 日志输出

该方法负责计算各个维度每个周期的耗时，进行日志输出。

```python
def log_stats(self, cur, start, end, total, ims_per_gpu=1, suffix='', log_all=False):
    """ Log the tracked statistics
    Args:
        cur (int)
        start (int)
        end (int)
        total (int)
        ims_per_gpu (int, optional): Defaults to 1.
        suffix (str, optional): Defaults to ''.
        log_all (bool, optional): Defaults to False.
    """
    if cur % self.logperiod == 0 or cur == end:
        eta_seconds = self.timers['iter'].average_time / ims_per_gpu * (end - cur)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        lines = [
            '[Testing][range:{}-{} of {}][{}/{}]'.format(start, end, total, cur, end - start + 1),
            '[{:.3f}s = {:.3f}s + {:.3f}s + {:.3f}s][eta: {}]'.format(
                *[self.timers[name].average_time / ims_per_gpu for name in self.default_timers], eta),
        ]

        if log_all:
            lines.append('\n|')
            for name, timer in self.timers.items():
                if name not in self.default_timers:
                    lines.append('{}: {:.3f}s|'.format(name, timer.average_time / ims_per_gpu))
        lines.append(suffix)
        logging_rank(''.join(lines))
```

#### 计时相关

```python
    def wait(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def add_timer(self, name):
        if name in self.timers:
            raise ValueError(
                "Trying to add a existed Timer which is named '{}'!".format(name)
            )
        timer = Timer()
        self.timers[name] = timer
        return timer

    def reset_timer(self):
        for _, timer in self.timers:
            timer.reset()

    def tic(self, name):
        if name == 'iter':
            self.iter += 1
        timer = self.timers.get(name, None)
        if not timer:
            timer = self.add_timer(name)
        timer.tic()

    def toc(self, name):
        timer = self.timers.get(name, None)
        if not timer:
            raise ValueError(
                "Trying to toc a non-existent Timer which is named '{}'!".format(name)
            )
        if self.iter > self.num_warmup:
            self.wait()
            return timer.toc(average=False)
```

这些方法控制计时：

- `wait` 启用pytorch的`torch.cuda.synchronize`函数，等待cuda操作全部完成以计时准确（synchronize函数注释：Waits for all kernels in all streams on a CUDA device to complete.）。
- `add_timer` 根据传入name添加相应计时器。在初始化函数中，默认添加`('iter', 'data', 'infer', 'post')`四个计时器。
- `reset_timer` 将所有的计时器重置。
- `tic` 根据传入name，开始新一轮计时。
- `toc` 根据传入name，计算当前轮次耗时。
