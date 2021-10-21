## 评估教程

### 介绍

Pet提供了两种评估模型的方法:

1. 根据需求，直接通过`tools/{type}/[project/subtask]/test_net.py`找到指定模型的测试脚本进行模型评估。  
* **type**:必填项。根据任务评估类型可选`vision`、`projects`、`tasks`。
* **project/subtask**:可选项。若`type == projects`,此处需指定具体项目名，例如：`crowdcounting`、`densepose`、`fairmot`等；若`type == projects`,此处需指定具体子任务名`action`、`cls`、`contrast`、`face`、`instanse`或者、`tddet`;若`type == vision`，此处无需填写。

2. 通过给`tools/test_net_all.py`传入配置项的方法指定具体的测试脚本位置`tools/{type}/[project/subtask]/test_net.py`并间接调用该脚本进行模型评估。

显然，`tools/test_net_all.py`是一种通用基础方法，遵循Pet基于配置项实例化对象的思想，我们通常基于第二种方法进行模型评估测试，可服务于各个算法框架。下面将从通用的模型评估测试脚本`tools/test_net_all.py`切入讲解模型评估的启动，进而以`tools/vision/test_net.py`作为具体用例介绍整套评估流程。


### 启动评估

使用`tools/test_net_all.py`评估模型时，可通过命令行参数指定具体的测试环境及配置文件等。具体有以下四个可选参数，一般情况下，常用的只有前两项甚至仅用第一项：
* **--cfg**：可指定一个YAML文件，该文件里包含了本次模型评估实验所需的一系列相关配置；注意，此处YAML的放置路径必须遵循[1.5.1 configs](https://github.com/BUPT-PRIV/Pet-doc/blob/dev/tutorials/configs_zh.md)中的标准，否则将会影响后续用于模型评估的测试脚本路径的定位。默认项：`cfgs/vision/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml`。
* **--gpu_id**：根据具体的运行环境,指定用于模型测试的GPU。默认项：`"0,1,2,3,4,5,6,7"`(8卡测试)。
* **--type**：测试类型，与`tools/{type}/[project/subtask]/test_net.py`中的`type`对应，用于进一步确定测试脚本路径，可选`vision`、`projects`、`tasks`。默认项：`None`，通常不需指定，可通过`--cfg`参数确定。
* **testing_script_args**：命令行所有剩余的参数，均转化为一个列表赋值给此项，如：TEST.RESIZE.SCALE 400。

根据配置项找到指定测试脚本的原理代码如下：
```python
# args.cfg_file 指向命令行--cfg参数,取值：cfgs/{type}/[type下的具体子项/...]/{XXXX.yaml}
testing_script = None
if args.type == "none":
    if "/vision" in args.cfg_file:
        testing_script = "tools/vision/test_net.py"
    elif "/projects" in args.cfg_file:
        project = args.cfg_file.split("/projects/")[-1].split("/")[0]
        testing_script = f"tools/projects/{project}/test_net.py"
    elif "/tasks" in args.cfg_file:
        subtask = args.cfg_file.split("/tasks/")[-1].split("/")[0]
        testing_script = f"tools/tasks/{subtask}/test_net.py"
else:
    if args.type == "vision":
        testing_script = "tools/vision/test_net.py"
    elif args.type == "projects":  # TODO
        project = args.cfg_file.split("projects/")[-1].split("/")[0]
        testing_script = f"tools/projects/{project}/test_net.py"
    elif args.type == "tasks":
        subtask = args.cfg_file.split("tasks/")[-1].split("/")[0]
        testing_script = f"tools/tasks/{subtask}/test_net.py"
assert testing_script is not None, "cfg path should be like cfgs/cnn/... or --type should be set"
```

调用指定的测试脚本的代码如下：
```python
cmd = [
    sys.executable,
    "-u", testing_script,
    f"--cfg={args.cfg_file}", # 对应testing_script可选的的命令行参数
    f"--local_rank={local_rank}",
]
cmd.extend(args.testing_script_args)
process = subprocess.Popen(cmd, env=current_env) # 执行python程序
```
注意：调用指定的测试脚本时，传入的参数要与测试脚本的可输入命令行参数匹配。

### 评估流程

每个具体项目的模型评估测试脚本都有一个主函数(main)和一个测试函数(test)。

主函数(main)主要用于根据配置项准备测试函数(test)所需的各个组件，包括数据集、加载器、模型、权重文件、测试钩子及测试引擎，并提供了一系列辅助计算，如：计算模型参数量、浮点运算次数、激活次数，属于模型测试的准备阶段；而测试函数则主要定义了抽象的模型测试逻辑，结合主函数的传入组件，展开具体的模型评估。

#### 一个简单例子

以最常用的视觉基础任务的测试评估脚本`tools/vision/test_net.py`为例，详细说明Pet的官方定义模型评估流程。此处只介绍模型评估测试的逻辑实现，具体函数应用细节请看API文档。

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

**(2) 创建分析器，计算模型参数量&浮点运算次数&激活次数**
```python
n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
if is_main_process() and cfg.ANALYSER.ENABLED:
    model = GeneralizedCNN(cfg) # 调用相应测试类型下的模型构建器，根据配置信息的模型超参数搭建模型
    model.eval()
    analyser = RCNNAnalyser(cfg, model, param_details=False) # 创建模型分析器对象
    n_params = analyser.get_params()[1] # 计算模型参数量
    conv_flops, model_flops = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="flops") # 计算卷积层浮点运算次数和模型前传浮点运算次数
    conv_activs, model_activs = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="activations") # 计算卷积层激活次数和模型前传激活次数
    del model # 删除模型
synchronize() # 使分布式训练时在所有进程之间同步的辅助函数 
```
该部分并非评估测试所必须，视情况可以省略

**(3) 搭建模型**
```python
# 核心调用：GeneralizedCNN，一个模型构建函数。根据配置信息构建网络结构，包括backbone、neck、global_haed等，并决定网络模块参数的状态(更新或冻结)
model = GeneralizedCNN(cfg)
logging_rank(model)
logging_rank(
    "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M"
    .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
)
```

**(4) 加载模型权重**
```python
# 创建模型CheckPointer类对象，并确定是否恢复最新训练的模型权重、优化器、学习率调节器等
test_weights = get_weights(cfg.MISC.CKPT, cfg.TEST.WEIGHTS) # 获取模型权重文件的路径
load_weights(model, test_weights) # 加载模型权重至第(3)步构建的模型中
model.eval()
model.to(torch.device(cfg.MISC.DEVICE))
```

**(5) 测试集读取，并构建数据加载器**

```python
# Create testing dataset and loader
dataset = build_dataset(cfg, is_train=False) # 根据配置信息，创建指定数据集类，读取数据
test_loader = make_test_data_loader(cfg, dataset) # 根据配置信息，创建指数据集加载器类，对读取的数据集进行迭代加载
```

**(6) 构建测试钩子(TestHook)**
```python
# 核心函数：build_test_hooks,根据配置信息，初始化返回TestHook类对象
all_hooks = build_test_hooks(args.cfg_file.split("/")[-1], log_period=10, num_warmup=0)
```

**(7) 构建测试引擎**
```python
# TestEngine 是模型测试的核心部件。由于不同任务对模型测试评估的处理方法差异较大，因此需要对测试评估方法进一步定义封装，根据实验配置决定具体实现的评估方法。
test_engine = TestEngine(cfg, model, dataset)
```

**(8) 开启测试流**
```python
# 通过上述步骤准备好测试所需组件后，调用评估测试函数test，开启正式测试
test(cfg, test_engine, test_loader, dataset, all_hooks)
```

* test函数

TestEngine 对不同任务的测试评估方法实现了封装，简化了test函数的实现。该部分不展开具体介绍，仅以代码注释加以讲解。

```python
def test(cfg, test_engine, loader, dataset, all_hooks):
    total_timer = Timer() 
    total_timer.tic() # 记录当前时刻
    all_results = [[] for _ in range(7)]
    with torch.no_grad():
        loader = iter(loader)
        for i in range(len(loader)):
            all_hooks.iter_tic()
            all_hooks.data_tic()
            inputs, targets, idx = next(loader)
            all_hooks.data_toc()

            all_hooks.infer_tic()
            eval_results = test_engine(inputs, idx, targets) # 核心部分：返回测试评估结果
            all_results = [results + eva for results, eva in zip(all_results, eval_results)]
            all_hooks.infer_toc()

            all_hooks.iter_toc()
            if is_main_process():
                all_hooks.log_stats(i, 0, len(loader), len(dataset)) # 记录测试结果

    all_results = list(zip(*all_gather(all_results)))
    all_results = [[item for sublist in results for item in sublist] for results in all_results]
    if is_main_process():
        total_timer.toc(average=False) # 记录测试总时间
        logging_rank("Total inference time: {:.3f}s".format(total_timer.average_time))
        torch.cuda.empty_cache()
        test_engine.close(all_results) 
```

模型测试评估结果的具体是现在TestEngine中，TestEngine 类通常封装在`pet/{type}/[type下的具体子项/...]/{core}/test.py`下，此处以`pet.vision.core.test.TestEngine`示例：
```python
class TestEngine(object):
    def __init__(self, cfg, model, dataset):

        ...
        self.processor = CNNPostProcessor(cfg, dataset) #
        ...

    def __call__(self, images, idx, targets=None):
        """
        Args:
            images (list[PIL.Image])

        Returns:
            list[ImageContainer]
        """
        self.images = images
        self.targets = targets

        if not self.cfg.MODEL.INSTANCE_ON: # 模型是否针对实例
            if self.cfg.MODEL.GLOBAL_HEAD.CLS_ON: # 模型是否用于分类
                self.global_cls_test() # 分类测试
            else:
                self.global_test()  # 包括semseg/panoseg/insseg/det测试
        else:
            self.ins_test()

        def roi_test():
            if len(self.features) > 0:  # instance has roi
                if self.cfg.MODEL.ROI_HEAD.GRID_ON:
                    self.ins_grid_test()
                if self.cfg.MODEL.ROI_HEAD.MASK_ON:
                    self.ins_mask_test()
                if self.cfg.MODEL.ROI_HEAD.KEYPOINT_ON:
                    self.ins_keypoints_test()
                if self.cfg.MODEL.ROI_HEAD.PARSING_ON:
                    self.ins_parsing_test()

        all_results = [[] for _ in range(7)]
        rois = self.result
        ins_num = sum(len(result_per_im) for result_per_im in rois)
        if ins_num > self.max_ins_per_gpu > 0:
            assert len(rois) == 1, "single image test"
            roi = rois[0]
            for i in range(0, ins_num, self.max_ins_per_gpu):
                self.result = [roi[i: i + self.max_ins_per_gpu]]
                roi_test()
                eval_results = self.processor(images, self.result, idx, targets)
                all_results = [results + eva for results, eva in zip(all_results, eval_results)]
        else:
            roi_test()
            all_results = self.processor(images, self.result, idx, targets)

        return all_results

        def close(self, all_results):
            self.processor.close(all_results)

        def ins_test(self):
            ...
        
        def global_cls_test(self):
            ...
        
        ...

```
