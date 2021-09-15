# Pet-doc-public
## 新-Pet文档框架

MM          [https://mmdetection.readthedocs.io/en/latest/]

PADLE     [https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html]

torch       [https://pytorch.org/vision/stable/index.html]

detecton2  [https://detectron2.readthedocs.io/en/latest/]

### 使用教程

    1.1 介绍

    1.2安装

    1.3快速开始

    1.4特征

    1.5使用

    ​	1.5.1 config 

    ​	1.5.2 model （与cfg联动、调用、registry）（曹朴）

    ​	1.5.3 data（dataloader、sampler、 transform 、 post_process) （马原东）

    ​	1.5.4 solver （曹朴）

    ​	1.5.5 training （朱）

    ​	1.5.6 evaluation （张）

    ​	1.5.7 inference （代）


    1.6 ModelZoo

    1.7 Benchmark

    * 1.8 工程部署

### API文档  (函数调用) 查询类

    2.1 pet.lib

    ​	2.2.1 config [https://detectron2.readthedocs.io/en/latest/modules/config.html]

    ​	2.2.2 backbone

    ​	2.2.3 data [https://detectron2.readthedocs.io/en/latest/modules/data.html]（马原东）

    ​		2.2.3.1 datasets

    ​		2.2.3.2 evaluation

    ​		2.2.3.3 samplers

    ​		2.2.3.4 structures

    ​		2.2.3.5 transforms

    ​		2.2.3.6 collate_batch

    ​	2.2.4 layers （基本都继承nn.Module，可以仿照pytorch文档格式写）（曹朴）

    ​	2.2.5 ops （这块非常多+杂）（2人）

    ​		2.2.5.1 loss（朱）

    ​		2.2.5.2 others （张）

    ​	2.2.6 utils （每个文件单开一个）（代）

    2.2 pet.cnn

    ​	2.2.1 core

    ​		2.2.1.1 inference

    ​		2.2.1.2 test

    ​	2.2.2 datasets

    ​		2.2.2.1 dataset

    ​		2.2.2.2 post_process

    ​		2.2.2.3 transform

    ​	2.2.3 utils （每个文件单开一个）

    ​	2.2.4 modeling

    ​		2.2.4.1 Generalized CNN （pet.cnn.modeling.model_builder.py)

    ​		2.2.4.2 Global xxx (每个任务的Module，比如GlobalCls）

### 应用实践   （[https://detectron2.readthedocs.io/en/latest/tutorials/models.html#build-models-from-yacs-config]

    3.1 快速开始

    3.2 基础教程 (三组公用) （跑试验+写）

    ​	3.2.1 Cls 张

    ​	3.2.2 Det 朱

    ​	3.2.3 Seg  代

    ​	3.2.4 Parsing （马原东）

    3.3.用户自定义使用（曹朴）

    ​	3.3.1 添加自定义数据集

    ​	3.3.2 添加自定义模型

    ​	*3.3.3 添加自定义模块~~

    ​	*3.3.4 添加自定义任务~~

    *3.4 Pet 模块开发*

    ​	~~3.4.1 提交issue~~

    ​	~~3.4.2 提交pr~~

    ​	~~……~~
