# 添加自定义数据集

Pet支持用户添加自定义数据集。对于已支持的数据格式，可以通过设置数据集相关参数简单地进行使用。若想添加目前不支持的数据格式，需要在Pet中添加对应的数据读取类。Pet已有的数据读取类和数据读取规范，可参考 [data](../../usage/data_zh.md) 文档。

本篇示例教程将添加一个遵循Pet支持格式的分类数据集。

## Example

本篇以二分类数据集为例，展示如何在Pet中支持新的数据集。

对于分类任务，Pet定义了 `ImageFolderDataset` 进行数据读取。数据图片按照相应类别放在对应文件夹下即可，无需额外的标注文件。

首先将数据集链接到Pet的数据目录下：

```bash
ln -s /path-to-dataset/ExampleDataet /path-to-pet/data/
```

### 数据格式

每个类别的图片各自放在一个文件夹下，目录结构如下所示：

```
ExampleDataset
	|--- Category_A
			|--- xxxxx.jpg
	|--- Category_B
			|--- xxxxx.jpg
```

### 数据集注册

在Pet中使用的数据集需要先注册，配置好数据目录以及相关信息。在使用时可以依靠配置好的设置快速启动。

数据集注册目录：`pet/lib/data/dataset_catalog.py`

```COMMON_DATASETS```字典存放了所有数据集信息，key为调用名称，用于在config中指定。在```COMMON_DATASETS```中添加如下信息：

```python
'example_dataset': {
        _DATASET_NAME: 'example',
        _IM_DIR: _DATA_DIR + 'ExampleDataset',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls',),
        _ANN_FIELDS: {
            'num_images': -1,
            'cls': {
                'num_classes': 2,
            },
        },
    },
```

### 配置文件指定数据集

在配置文件中需要指定数据读取方式以及使用的数据集。本文所演示的数据用 `pet.lib.data.datasets.image_folder_dataset.ImageFolderDataset` 进行数据读取，Pet支持的全部数据读取方式如下：

```python
DATASET_TYPES = {
    "cifar_dataset": CifarDataset,
    "coco_dataset": COCODataset,
    "coco_instance_dataset": COCOInstanceDataset,
    "image_folder_dataset": ImageFolderDataset,
}
```

在配置文件中需要添加两个字段：

```yaml
DATA:
  DATASET_TYPE: "image_folder_dataset"
TRAIN:
  DATASETS: ("example_dataset",)
```

`DATASET_TYPE` 仅可从上面`DATASET_TYPES`的keys中选择，Pet运行时会根据字符串调用对应的数据读取类。

`TRAIN.DATSETS` 的字段与`pet/lib/data/dataset_catalog.py`定义的一致。

训练过程可参考 examples/train_cls.md 。

## 其他

如果想在Pet上支持其他数据格式，可以参考 tutorials/data.md ，在与已有的数据读取类风格和接口保持一致的基础上，自定义一个数据读取类，并在 `pet/vision/datasets/dataset.py` 中的`DATASET_TYPES`中添加对其的支持与导入。