# 配置系统

配置系统是Pet在训练和测试过程中，用户对各个环节进行配置的控制台。配置系统使得Pet运行过程中的设置变得灵活、高效和容易实现，同时配置系统还为pet提供了可扩展的配置，使得Pet用户能够自定义配置，不仅如此，配置系统还包括以下优点。

- 配置设置易读性：由于配置系统是基于YAML文件类型，YAML文件具备可读性高、和脚本语言交互性好以及扩展性高等优势，所有训练测试过程中的配置信息都存放在YAML文本文件中，可以使用文本编辑器直接修改和设置相应配置细节。
- 更新的便捷与即时性：由于配置系统具有参数配置高度集成的特点，在配置系统中某些配置被更改后，无需再逐一修改训练与测试中的脚本文件，大大提升了用户在优化迭代及测试过程中的效率。
- 可扩展性：配置系统具有很强的扩展性，用户可根据自身需求在配置系统中添加所需参数，同时利用配置系统，用户能够自定义配置细节。

我们使用YAML文件将训练和测试过程中的参数以键值对的形式进行保存，同时针对Pet的不同组件以及各任务中的具体操作参数使用层级关系进行对应。在配置文件书写过程中需注意的是YAML对大小写敏感，需严格注意Pet提供参数的大小写；对于每个对象的格式表达为key: value，同时冒号后面要加一个空格。

在代码中，所有的配置通过 ```yacs.config.CfgNode``` 类记录。配置的默认值在 ```pet/lib/config``` 中设置，**请注意添加配置时需先在该文件中设置默认值**。

## 修改配置

为保证Pet的统一，默认配置设置完成后，如无特殊原因将不会修改。在模型使用中，若想修改配置，可以通过 ```yaml``` 文件和终端指令修改配置。

**示例：**

- yaml文件修改配置：`MODEL.BACKBONE = 'resnet'`

  ```
  MODEL:
    BACKBONE: "resnet"
  ```

- 终端指令修改配置：`TRAIN.BATCH_SIZE = 1`

  ```
  python tools/train_net_all.py --cfg /path/to/cfg TRAIN.BATCH_SIZE 1
  ```

## 配置系统的结构与内容

默认配置文件的目录如下：

```
pet.lib.config
	|--- __init__.py
	|--- config.py
	|--- data.py
	|--- solver.py
	|--- model/
		|--- backbone.py
		|--- neck.py
		|--- head.py
		|--- model.py
```

根据配置的分类，在相应的文件中进行设置。

配置文件的一级字段包含一下内容：

- MISC：基础配置
- ANALYSER：analyser相关配置
- EVAL：评估阶段配置
- VIS：可视化配置
- DATA：数据相关配置
- TRAIN：训练配置
- TEST：测试配置
- SOLVER：优化器及调度器配置
- MODEL：模型配置

### 通用配置 config.py

#### MISC

| 字段         | 类型   | 默认值 | 备注                                              |
| ------------ | ------ | ------ | ------------------------------------------------- |
| MISC.VERSION | string | "0.7a" | Pet版本                                           |
| MISC.CKPT    | string | ""     | checkpoints和loggersf的目录                       |
| MISC.DEVICE  | string | "cuda" | 使用设备。若为"cuda"则使用GPU，若为"cpu"则使用CPU |

（以此类推）

## 配置关联

为便捷使用，Pet提供了配置之间的判断关联，可以设置某些字段为特定值时，调整另一配置。配置的交互部分在 ```pet/lib/config/__init__.py``` 的 ```infer_cfg``` 函数中。

**示例：**

- 当语义分割中某一具体模型启用时，启用语义分割Head：

  ```python
      if (cfg.MODEL.GLOBAL_HEAD.SEMSEG.AUXSEG_ON
          or cfg.MODEL.GLOBAL_HEAD.SEMSEG.FUSEDNET_ON
          or cfg.MODEL.GLOBAL_HEAD.SEMSEG.MASKFORMER_ON
          or cfg.MODEL.GLOBAL_HEAD.SEMSEG.PFPNNET_ON
          or cfg.MODEL.GLOBAL_HEAD.SEMSEG.PSPNET_ON
          or cfg.MODEL.GLOBAL_HEAD.SEMSEG.SEMSEGFPN_ON):  # noqa: E129
          cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON = True
  ```

- 当某一类型可视化启用时，开启可视化模块：

  ```python
      if (cfg.VIS.BOX.ENABLED
          or cfg.VIS.CLASS.ENABLED
          or cfg.VIS.KEYPOINT.ENABLED
          or cfg.VIS.MASK.ENABLED
          or cfg.VIS.PANOSEG.ENABLED
          or cfg.VIS.PARSING.ENABLED
          or cfg.VIS.SEMSEG.ENABLED):  # noqa: E129
          cfg.VIS.ENABLED = True
  ```

  

