# 模型构建

## 模型

Pet中视觉模型分为四个部分：

- backbone
- neck
- global head
- roi head

其中，backbone是必须的，其余三个部分是可选的。根据不同任务以及模型结构可以进行相对自由的组合，并且可以仅通过配置文件进行控制，这也是Pet平台统一代码的好处。所有的模型类全部继承`torch.nn.Module`，以保证模型的构建，前向等保持一致。

Pet中的模型是逐级构建的，在视觉任务中，最上层是完整的模型类`GeneralizedCNN` 。该类组合并实例化了模型的所有部分，并且在前向函数中控制数据流。构建及前向过程如图所示（以ResNet+FCOS为例）：

<p align="center"><img width="60%" src="../img_sources/330-430.png" /></p>



### GeneralizedCNN类

继承 `nn.Module`，在文件`pet/cnn/modeling`中

这个类用于组合 Backbone, Neck, Global Head, Roi Head，在构造函数中**根据配置文件**实例化对应模型，在forward函数中按顺序控制数据传输。

#### 构造函数

在```GeneralizedCNN```的构造过程中，会根据配置（传入的```cfg```参数，类型是```CfgNode```），实例化模型的四个部分（部分可选）。

由于模型的Head结构与任务高度相关，所以Head部分多进行一层封装。对同类型的任务封装为一个任务相关的类，如分类模型的Head为```GlobalCls```类。而Backbone和Neck部分较为统一或独立，所以不需要统一地将其分类再额外封装。Backbone和Neck通过registry机制（详见本文最后一章）进行实例化，根据```cfg.MODEL.BACKBONE```和```cfg.MODEL.NECK``` 指定使用的模型。具体可参考：Backbone，Neck。（待补单独文章）

```python
from pet.cnn.modeling import backbone, neck  # noqa: F401
from pet.cnn.modeling.global_head.cls.cls import GlobalCls
from pet.cnn.modeling.global_head.det.det import GlobalDet
from pet.cnn.modeling.global_head.insseg.insseg import GlobalInsSeg
from pet.cnn.modeling.global_head.panoseg.panoseg import GlobalPanoSeg
from pet.cnn.modeling.global_head.semseg.semseg import GlobalSemSeg
from pet.cnn.modeling.roi_head.cascade.cascade import ROICascade

class GeneralizedCNN(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        # Backbone
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        # Neck
        if cfg.MODEL.NECK:
            Neck = registry.NECKS[cfg.MODEL.NECK]
            self.neck = Neck(cfg, dim_in, spatial_in)
            dim_in = self.neck.dim_out
            spatial_in = self.neck.spatial_out
            #  Global Head: Classification
            if cfg.MODEL.GLOBAL_HEAD.CLS_ON:
                self.global_cls = GlobalCls(cfg, dim_in, spatial_in)
```

#### 初始化

```GeneralizedCNN```定义了初始化函数``` _init_modules ```用于冻结模型参数。

```python
 def _init_modules(self):
    if self.cfg.MODEL.FREEZE_BACKBONE:
        for p in self.backbone.parameters():
            p.requires_grad = False
    if self.cfg.MODEL.FREEZE_NECK and self.cfg.MODEL.NECK:
        for p in self.neck.parameters():
            p.requires_grad = False
```

#### 前向函数

```GeneralizedCNN```在```forward```函数中，连接了模型的不通模块用于前向推理。需要注意不同模块之间的数据流，保持上一个模型的输出与下一个模型的输入定义的一致性。

```python
def forward(self, images, targets=None):
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")

    # Backbone
    conv_features = self.backbone(images.tensor)

    # Neck
    if self.cfg.MODEL.NECK:
        conv_features = self.neck(conv_features)

    # Global Head
    cls_losses = {}
    if self.cfg.MODEL.GLOBAL_HEAD.CLS_ON:
        clses, conv_features, cls_losses = self.global_cls(images, conv_features, targets)

    semseg_losses = {}
    if self.cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON:
        semsegs, conv_features, semseg_losses = self.global_semseg(images, conv_features, targets)

    panoseg_losses = {}
    if self.cfg.MODEL.GLOBAL_HEAD.PANOSEG_ON:
        panosegs, panoseg_losses = self.global_panoseg(images, conv_features, targets)
    
    # 省略其他head判断
    
    if self.training:
        outputs = {'metrics': {}, 'losses': {}}
        outputs['losses'].update(cls_losses)
        outputs['losses'].update(proposal_losses)
        outputs['losses'].update(semseg_losses)
        outputs['losses'].update(panoseg_losses)
        outputs['losses'].update(insseg_losses)
        outputs['losses'].update(roi_losses)
        return outputs

    return result
```

根据是否处于训练阶段，```forward```的返回值是不一样的。训练阶段会返回一个字典，形式如下所示：

```python
{
	'metrics': {},
	'losses': {
		'loss1': torch.Tensor,
		'loss2': torch.Tensor,
	}
}
```

```losses```包含模型产生的所有损失。并且这些损失**已经在前面的过程（Head模型中）加权**，在后续反向传播中，会直接将这个字典中所有的损失相加作为模型总体损失来回传梯度。

在非训练阶段（测试、预测），```GeneralizedCNN```会返回模型结果。如分类任务的类别，检测任务的检测框。

### Head

模型的Head模块根据任务进行划分。在每个任务的Head类中，根据配置文件，调用具体算法或模型的Module。

以目标检测任务为例，在`GeneralizedCNN`中，根据配置文件实例化`GlobalDet`类：

```python
#  Global Head: Detection
if cfg.MODEL.GLOBAL_HEAD.DET_ON:
    self.global_det = GlobalDet(cfg, dim_in, spatial_in)
```

在`GlobalDet`的构造函数中根据配置文件，实例化具体的网络模型，如RetinaNet, FCOS等。以FCOS算法为例，在构造函数中根据配置文件实例化`FCOSModule`，并在前向函数中根据配置文件调用`FCOSModule`进行前向运算。

```python
class GlobalDet(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        # .......
        # FCOS
        if self.cfg.MODEL.GLOBAL_HEAD.DET.FCOS_ON:
            self.fcos = FCOSModule(self.cfg, dim_in, spatial_in)
    def forward(self, images, conv_features, targets=None, image_result=None):
		# .......
		if self.cfg.MODEL.GLOBAL_HEAD.DET.FCOS_ON:
            fcos_proposals, fcos_loss = self.fcos(images, conv_features, targets)
            det_results["fcos_proposals"] = fcos_proposals
            det_losses.update(fcos_loss)
            if self.cfg.MODEL.GLOBAL_HEAD.DET.AUXDET_ON and self.training:
                _, aux_loss = self.aux_det(images, conv_features, targets, self.fcos)
                det_losses.update(aux_loss)
		# ......
```

### FCOSModule

`FCOSModule`在`pet/vision/modeling/global_head/det/fcos/fcos.py`中，该代码还定义了`FCOSHead`类。

Head和Module的区别是，Head类负责搭建模型网络，进行计算。而Module中调用Head类计算后，根据状态（是否处于训练模式）来计算loss或对网络输出后处理，输出所需要的结果。

module的forward函数节选：

```python
def forward(self, features):
	# ......

	cls_logits, bbox_preds, extra_preds = self.head(features)
		
	# 根据是否训练调用不同方法
	if self.training:
		return self._forward_train(locations, cls_logits, bbox_preds, extra_preds, images.image_sizes, targets, anchors)
	else:
		return self._forward_test(locations, cls_logits, bbox_preds, extra_preds, images.image_sizes, anchors)
		
# 训练阶段返回loss，其他阶段返回检测框
def _forward_train(self, locations, cls_logits, bbox_preds, extra_preds, targets):
    proposals = None
    losses = self.loss_evaluator(locations, cls_logits, bbox_preds, extra_preds, targets)
    return proposals, losses

def _forward_test(self, locations, cls_logits, bbox_preds, extra_preds, image_sizes):
    image_results = self.box_selector(locations, cls_logits, bbox_preds, extra_preds, image_sizes)
    return image_results, {}
```

## Registry

Registry是模型构建中的重要机制，该类继承字典类，定义在`pet/lib/utils/registry.py`中。

通过Registry，可以方便的创建模型字典，使得在构建模型的时候，可以通过读取配置文件中的模型名称来调用相应的类。

使用时，需要在`pet/cnn/modeling/registry.py`中先实例化对应的Registry类，如：

```python
from pet.lib.utils.registry import Registry
BACKBONES = Registry()
```

在定义模型类时，通过装饰器将其加入字典，如：

```python
@registry.BACKBONES.register("resnet")
def resnet(cfg):
    stride = cfg.MODEL.RESNET.STRIDE
    assert stride in [8, 16, 32]
    model = ResNet(cfg, stride=stride)
    return model
```

在使用时，可以通过读取配置文件中的模型名称来调用模型类，如：

```python
Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
```

对应配置文件中，`MODEL.BACKBONE = 'resnet'` 。