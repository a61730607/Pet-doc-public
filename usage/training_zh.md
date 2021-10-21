## 训练教程

### 介绍

Pet提供了两种训练模型的方法:

1. 根据需求，直接通过`tools/{type}/[project/subtask]/train_net.py`找到指定模型的训练脚本进行训练。  
* **type**:必填项。根据训练类型可选`vision`、`projects`、`tasks`。
* **project/subtask**:可选项。若`type == projects`,此处需指定具体项目名，例如：`crowdcounting`、`densepose`、`fairmot`等；若`type == projects`,此处需指定具体子任务名`action`、`cls`、`contrast`、`face`、`instanse`或者、`tddet`;若`type == vision`，此处无需填写。

2. 通过给`tools/train_net_all.py`传入配置项的方法指定具体的训练脚本位置`tools/{type}/[project/subtask]/train_net.py`并间接调用该脚本进行训练。

显然，`tools/train_net_all.py`是一种通用基础方法，遵循Pet基于配置项实例化对象的思想，我们通常基于第二种方法进行模型训练，可服务于各个算法框架的训练。下面将从通用训练脚本`tools/train_net_all.py`切入讲解模型训练的启动，进而以`tools/vision/train_net.py`作为具体用例介绍整套训练流程。


### 启动训练

使用`tools/train_net_all.py`训练模型时，可通过命令行参数指定具体的训练环境及配置文件等。具体有以下四个可选参数，一般情况下，常用的只有前两项甚至仅用第一项：
* **--cfg**：可指定一个YAML文件，该文件里包含了所有训练时使用到的可以调节的超参数；注意，此处YAML的放置路径必须遵循[配置系统](./configs_zh.md)中的标准，否则将会影响后续的训练脚本路径定位。默认项：`cfgs/vision/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml`。
* **--gpu_id**：根据具体的运行环境,指定用于训练的GPU。默认项：`"0,1,2,3,4,5,6,7"`(8卡训练)。
* **--type**：训练类型，与`tools/{type}/[project/subtask]/train_net.py`中的`type`对应，用于进一步确定训练脚本路径，可选`vision`、`projects`、`tasks`。默认项：`None`，通常不需指定，可通过`--cfg`参数确定。
* **training_script_args**：剩余的不属于任何训练类型的训练项目脚本。

根据配置项找到指定训练脚本的原理代码如下：
```python
# args.cfg_file 指向命令行--cfg参数,取值：cfgs/{type}/[type下的具体子项/...]/{XXXX.yaml}
if args.type == "none":
    if "cfgs/vision" in args.cfg_file:
        training_script = "tools/vision/train_net.py"
    elif "cfgs/projects" in args.cfg_file:
        project = args.cfg_file.split("cfgs/projects/")[-1].split("/")[0]
        training_script = "tools/projects/{}/train_net.py".format(project)
    elif "cfgs/tasks" in args.cfg_file:
        subtask = args.cfg_file.split("cfgs/tasks/")[-1].split("/")[0]
        training_script = "tools/tasks/{}/train_net.py".format(subtask)
else:
    if args.type == "vision":
        training_script = "tools/vision/train_net.py"
    elif args.type == "projects":
        project = args.cfg_file.split("projects/")[-1].split("/")[0]
        training_script = "tools/projects/{}/train_net.py".format(project)
    elif args.type == "tasks":
        subtask = args.cfg_file.split("tasks/")[-1].split("/")[0]
        training_script = "tools/tasks/{}/train_net.py".format(subtask)
```

调用指定的训练脚本的代码如下：
```python
cmd = [
            sys.executable, # sys.executable返回python解释器路径 /.../python.exe
            "-u", training_script, 
            f"--cfg={args.cfg_file}", # 对应training_script可选的的命令行参数
            f"--local_rank={local_rank}",
        ]
cmd.extend(args.training_script_args)

process = subprocess.Popen(cmd, env=current_env) # 执行python程序
```
注意：调用指定的训练脚本时，传入的参数要与训练脚本的可输入命令行参数匹配。

### 训练流程

每个具体项目的训练脚本都有一个主函数(main)和一个训练函数(train或run)。

主函数(main)主要用于根据配置项准备训练函数(train或run)所需的各个组件，包括数据集、加载器、模型、权重文件、优化器、学习率调节器及各种钩子(HOOK)，并提供了一系列辅助计算，如：计算模型参数量、浮点运算次数、激活次数，属于训练的准备阶段；而训练函数则主要定义了抽象的训练逻辑，结合主函数的传入组件，展开具体的模型训练。

#### 一个简单例子

以最常用的视觉基础任务训练脚本`tools/vision/train_net.py`为例详细说明Pet的官方定义训练流程。此处只介绍训练逻辑，具体函数应用细节请看API文档。

* main函数

开启一个训练任务之前，需要进行如下步骤：

**(1) 读取并融合配置文件信息**
```python
cfg = get_base_cfg() # 获取原始默认配置参数
cfg.merge_from_file(args.cfg_file) # args.cfg_file是--cfg指定的yaml配置文件，通过merge_from_file这个函数会将yaml文件中指定的超参数对原始配置默认值进行覆盖
cfg.merge_from_list(args.opts) # 作用同上面的类似，只不过是通过命令行的方式覆盖
cfg = infer_cfg(cfg, args.cfg_file) 
cfg.freeze() # freeze函数的作用是将超参数值冻结，避免被程序不小心修改
```
配置信息优先级：命令行＞配置文件＞默认配置

**(2) 分析器类初始化，计算模型参数量&浮点运算次数&激活次数**
```python
n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
if is_main_process() and cfg.ANALYSER.ENABLED:
    model = GeneralizedCNN(cfg) # 调用相应训练类型下的模型构建器，根据配置信息的模型超参数搭建模型
    model.eval()
    analyser = RCNNAnalyser(cfg, model, param_details=False) # 创建模型分析器对象
    n_params = analyser.get_params()[1] # 计算模型参数量
    conv_flops, model_flops = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="flops") # 计算卷积层浮点运算次数和模型前传浮点运算次数
    conv_activs, model_activs = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="activations") # 计算卷积层激活次数和模型前传激活次数
    del model # 删除模型
synchronize() # 使分布式训练时在所有进程之间同步的辅助函数 
```
该部分并非训练所必须，视情况可以省略

**(3) 模型类初始化，搭建模型**
```python
# 核心调用：GeneralizedCNN，一个模型构建函数。根据配置信息构建网络结构，包括backbone、neck、分类层等，并决定网络模块参数的状态(更新或冻结)
model = GeneralizedCNN(cfg)  
logging_rank(model)
logging_rank(
    "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | Activations: {:.4f}M / Conv_Activations: {:.4f}M"
    .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
)
```

**(4) CheckPointer类初始化，加载预训练权重或随机初始化模型权重**
```python
# 创建模型CheckPointer类对象，并确定是否恢复最新训练的模型权重、优化器、学习率调节器等
checkpointer = CheckPointer(cfg.MISC.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME) 

model = checkpointer.load_model(model, convert_conv1=cfg.MISC.CONV1_RGB2BGR) # 加载预训练权重或随机初始化模型权重
model.to(torch.device(cfg.MISC.DEVICE)) # 把模型加载到指定设备上
if cfg.MISC.DEVICE == "cuda" and cfg.MISC.CUDNN:
    cudnn.benchmark = True
    cudnn.enabled = True
```

**(5) 优化器类初始化**
```python
optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build() # 根据配置文件关于优化器类型、学习率、权重衰减等参数初始化一个优化器对象

# 根据CheckPointer类初始化设置决定是否用最新训练的模型优化器状态以及所使用的超参数的信息来覆盖上述初始化优化器
optimizer = checkpointer.load_optimizer(optimizer) 
logging_rank("The mismatch keys: {}".format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))))
```
关于CheckPointer类加载优化器的代码如下：
```python
def load_optimizer(self, optimizer):
    if self.resume:
        optimizer.load_state_dict(self.checkpoint.pop('optimizer')) # 恢复优化器状态以及所使用的超参数的信息
        logging_rank('Loading optimizer done.')
    else:
        logging_rank('Initializing optimizer done.')
    return optimizer
```

**(6) 读取数据，并构建数据加载器**

```python
# Create training dataset and loader
dataset = build_dataset(cfg, is_train=True) # 根据配置信息，创建指定数据集类，读取数据
start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter) # 根据配置信息，创建指数据集加载器类，对读取的数据集进行迭代加载
max_iter = len(train_loader) # 此处以iter为周期进行训练，整个训练的总迭代次数等于训练数据加载次数
iter_per_epoch = max_iter // cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS # 每个epoch的迭代次数等于总迭代次数除以设置的epoch个数
```
注意，除了上述以**iter**为周期的训练模式，Pet还有以**epoch**为周期的训练模式，其代码设置如下：
```python
iter_per_epoch = len(train_loader)
...
max_iter = iter_per_epoch
```
两种训练模式的区别：
* **iter**：整个训练过程中，每个样本只加载训练过一次，所有样本数据加载完一次，即完成训练
* **epoch**：所有样本数据加载完一次视为一个epoch，即有多少个epoch，每个样本在整个训练过程中就被加载过几次

对应的两种模式的训练函数实现逻辑也会有所不同。

**(7) 学习率调节器类初始化**
```python
# 和创建优化器类似，先根据配置信息及优化器初始化一个学习率调节器类对象，再根据CheckPointer类初始化设置决定是否覆盖学习率调节器状态及超参数信息
scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
scheduler = checkpointer.load_scheduler(scheduler)
```

**(8) 模型分布式加载**
```python
distributed = get_world_size() > 1 # 若GPU个数是大于1，则将模型通过DDP模式进行多GPU分布式训练
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )
```

**(9) 构建训练所需钩子(HOOK)**
```python
# 核心函数：build_train_hooks,根据配置信息及上述步骤创建的各组件信息，构建训练过程中所需的一系列钩子类对象，并以列表形式返回，用于训练函数遍历
if cfg.SOLVER.SCHEDULER.WARM_UP_EPOCHS is None:
    warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_ITERS
else:
    warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_EPOCHS * iter_per_epoch
all_hooks = build_train_hooks(
    cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False, precise_bn_args=precise_bn_args
)
```
钩子(HOOK)类的具体介绍请看[Hook](./hook_zh.md)

**(10) 开启训练流**
```python
# 通过上述步骤准备好训练所需组件后，调用训练函数train，开启正式训练
train(cfg, model, train_loader, optimizer, scheduler, checkpointer, all_hooks)
```

* train函数

Pet的训练工作流与基于pytorch的训练工作流类似，只是将部分默认操作和用户自定义操作借助HOOK机制加以实现，当train函数运行到预定义的位点时候就会调用对应Hook中的方法([Hook](./hook_zh.md))。因此该部分不展开具体介绍，仅以代码注释加以讲解。

```python
def train(cfg, model, loader, optimizer, scheduler, checkpointer, all_hooks):
    
    # switch to train mode
    """
    model.train() ：启用 BatchNormalization 和 Dropout
    model.eval() ：不启用 BatchNormalization 和 Dropout
    """
    model.train()

    # main loop
    start_iter = scheduler.iteration

    iteration = start_iter
    max_iter = len(loader)
    iter_loader = iter(loader)
    logging_rank("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter=start_iter, log_period=cfg.MISC.DISPLAY_ITER) as storage:
        try:
            '''
            HOOK: before_train():开始训练前调用
            '''
            for h in all_hooks:  # all_hooks：一个保存了默认HOOK对象和根据配置用户定义的HOOK对象的list
                h.before_train() #hooks
            for iteration in range(start_iter, max_iter + 1):
                '''
                HOOK: before_step():开始一次迭代前调用
                '''
                for h in all_hooks:
                    h.before_step(storage=storage) #hooks

                data_start = time.perf_counter() # 记录数据开始加载时间
                inputs, targets, _ = next(iter_loader)

                inputs = inputs.to(cfg.MISC.DEVICE)
                targets = [target.to(cfg.MISC.DEVICE) for target in targets]
                data_time = time.perf_counter() - data_start # 计算数据加载时间

                optimizer.zero_grad() #梯度清零

                outputs = model(inputs, targets) #输入数据，模型前向传播，返回预测值
                losses = sum(loss for loss in outputs["losses"].values())
                metrics_dict = outputs["losses"] # metrics_dict：指标字典，保存训练过程的状态，包括loss，数据加载时间等
                metrics_dict["data_time"] = data_time
                if cfg.TRAIN.METRICS_ON:
                    write_metrics(metrics_dict, storage)
                losses.backward() # losses反传

                #由于权重衰减，如果我们不在这里手动设置grad=None，feature_adapt中的权重将衰减为零
                if (cfg.MODEL.AUXDET.STEP1_ENABLED
                    and iteration <= cfg.MODEL.AUXDET.DISTANCE_LOSS_WARMUP_ITERS):  # noqa: E129
                    for p in model.module.global_det.aux_det.feature_adapt.parameters():
                        p.grad = None

                optimizer.step() #梯度更新

                '''
                HOOK: after_step():经过一次迭代后调用
                '''
                for h in all_hooks:
                    h.after_step(storage=storage)

                if is_main_process():
                    # Save model
                    if cfg.SOLVER.SNAPSHOT_ITER > 0 and iteration % cfg.SOLVER.SNAPSHOT_ITER == 0:
                        checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix="iter")
                storage.step()
        finally:
            if is_main_process():
                if iteration % cfg.SOLVER.SNAPSHOT_ITER != 0 and (iteration > 1000 or iteration == max_iter):
                    checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix="iter")

            '''
            HOOK: after_train():完成训练后调用
            '''
            for h in all_hooks:
                h.after_train(storage=storage)
```

通过举例介绍Pet的工作流可以发现，Pet的整个训练工作流依旧遵循“逻辑抽象+基于配置项实例化对象”的思想，因此，在使用Pet训练模型的过程中，用户可以通过自定义配置信息，实现样化的训练流。此外，再次验证了HOOK机制对训练的可扩展性强。