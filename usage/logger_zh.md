# 日志系统

日志模块是模型训练与测试阶段的辅助模块，它可以在控制台中输出信息以帮助您在训练与测试模型时能够监控模型的学习状态。

日志相关

### EventWriter

Writer的基类模型，定义了write和close方法。其中write方法是子类必须重写的方法，否则会报错。

```python
class EventWriter(object):
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass
```

基于EventWriter，Pet定义了三个写入类：

- JSONWriter：写入JSON文件；
- TensorboardXWriter：写入Tensorboard日志；
- CommonMetricPrinter：控制台输出及保存日志。

#### JSONWriter

JSON写入类，传入json文件路径以及日志保存频率。

在构造函数中，根据传入的json文件路径创建文件句柄`self.file_handle`。在`write`函数中，通过文件句柄写入通过`json.dumps`转换为字符串格式的字典，并通过`flush()`方法刷新。在`close`函数中，调用`self.file_handle.close()`释放文件。

```python
class JSONWriter(EventWriter):
    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int, optional): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True. Defaults to 20.
        """
        self.file_handle = open(json_file, "a")
        self.window_size = window_size

    def write(self, *args, **kwargs):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter + 1}
        to_save.update(storage.latest_with_smoothing_hint(self.window_size))
        self.file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self.file_handle.flush()
        try:
            os.fsync(self.file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self.file_handle.close()
```

#### TensorboardXWriter

Tensorboard写入。在`write`方法中，调用`torch.utils.tensorboard.SummaryWriter`类的`add_scalar`方法，写入`key`、`value`和`iteration`。

```python
class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, **kwargs)

    def write(self, **kwargs):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self.window_size).items():
            self.writer.add_scalar(k, v, storage.iter)

    def close(self):
        if hasattr(self, "writer"):  # doesn't exist when the code fails at import
            self.writer.close()
```

#### CommonMetricPrinter

CommonMetricPrinter类用于在控制台输出日志结果。

```python
class CommonMetricPrinter(EventWriter):
    """
    Print __common__ metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, yaml, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.max_iter = max_iter
        self.yaml = yaml

        log_path = os.path.join(yaml, "log")
        filename = f"train_{os.environ.get('CURRENT_TIME')}.txt"
        logger = logging.getLogger("Training")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        plain_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
        self.logger = setup_logging(log_path, filename=filename, logger=logger, formatter=plain_formatter)

    def write(self, epoch, max_epoch, **kwargs):
        storage = get_event_storage()
        iteration = storage.iter

        data_time, time, metrics = None, None, {}
        eta_string = "N/A"
        try:
            data_time = storage.history("data_time").avg(20)
            time = storage.history("time").global_avg()
            if max_epoch is not None:
                eta_iter = max_epoch * self.max_iter - iteration - 1
                iteration = iteration % self.max_iter
            else:
                eta_iter = self.max_iter - iteration
            eta_seconds = time * (eta_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            for k, v in storage.latest().items():
                if "acc" in k:
                    metrics[k] = v
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        losses = [
            "{}: {:.6f}".format(k, v.median(20))
            for k, v in storage.histories().items()
            if "loss" in k and "total" not in k
        ]
        skip_losses = len(losses) == 1
        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        lines = """\
|-[{yaml}]-{epoch}[iter: {iter}/{max_iter}]-[lr: {lr}]-[eta: {eta}]
                 |-[{memory}]-[{time}]-[{data_time}]
                 |-[total loss: {total_loss}]{losses}
\
""".format(
            yaml=self.yaml.split('/')[-1] + '.yaml',
            eta=eta_string,
            iter=iteration + 1,  # start from iter 1
            epoch='' if epoch is None else '[epoch: {}/{}]-'.format(epoch, max_epoch),
            max_iter=self.max_iter,
            lr=lr,
            memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            time="iter_time: {:.4f}".format(time) if time is not None else "iter_time: N/A",
            data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
            total_loss="{:.4f}".format(storage.histories()["total_loss"].median(20)),
            losses="-[losses]-[{}]".format("  ".join(losses)) if not skip_losses else "",
        )

        if len(metrics):
            lines += """\
                 {metrics}\
""".format(metrics="|" + "".join(
                ["-[{}: {:.4f}]".format(k, v) for k, v in metrics.items()]
            )
           )
        else:
            lines = lines[:-1]
        logging_rank(lines, logger=self.logger)
```