## Changelog


### v0.7a (12/09/2021)

**Improvements**
- Code style standardization.

**New Features**
- Add BEiT/CSwin/MiT/PCPVT/PVT/Swin/T2T/Vit/VOLO backbones.
- Add eqlv2/fed loss for long-tailed recognition.
- Add centernet2.
- Add ota.
- Add maskformer.
- Add more features to fcos.

### v0.6 (01/03/2021)

**Improvements**
- Code style standardization.
- Adjust model builder, split it into Global/RoI head.
- Build ImageContainer to remove dependence of Boxlist for other tasks.
- Make sure all InsSeg tasks have box results.
- Unify ops of each task.
- Iterative upgrade related code of projects.

**New Features**
- Add GhostNet support to pet/lib/backbone.
- Move pet/projects/face to pet/tasks/face.
- Add tddet to pet/tasks/tddet.
- Add tddfa to pet/projects/tddfa.
- Add more features to fcos.

### v0.5 (23/09/2020)

**Improvements**
- Support Pytorch 1.5 to 1.6.
- Sort imports.
- Standardize the format of input and output (head/output).
- Clean up and simplify cfgs.
- Standardize transforms.

**New Features**
- Add more internal parameters (spatial_in/spatial_out) of modules.
- Move pet/lib/ops boxlist_ops to pet/lib/layers.
- Remove dependency on Apex。

### v0.4 (30/08/2020)

**Improvements**
- Support Pytorch 1.3 to 1.5.
- Fast and accurate training / testing.
- New cfg call logic.
- New pet/projects rule.
- Code style standardization.

**Notation**
- This version of pet/rcnn is not compatible with the previous versions.
- The dtype of target is change in SigmoidFocalLoss() from int to long.

**New Features**
- Support pet/semseg.
- New training and test data stream.
- Rename pet/models as pet/lib/backbone.
- Move pet/utils to pet/lib/utils.
- Move pet/utils/data to pet/lib/datasets.
- Move pet/face to pet/projects/face.
- Move pet/rcnn densepose and hier to pet/projects
- Move and clean up pet/models/ops to pet/lib/ops.
- Refactoring make_conv, make_norm, and add get_conv_op, get_norm_op, mask_act, make_ctx.
- Add pet/lib/layer for some base classes.
- Add ResNeSt, RegNet, SpineNet support to pet/lib/backbone.
- Add HSD support to pet/ssd.
- Add CenterNet, SOLOv2, CondInst, PolarMask, Guided Anchoring, EQL loss support to pet/rcnn.
- Add BiSeNetv1, ASPPV3p, SemSegFPN, OCRNet to pet/semseg and support more backbone and dataset.
- Add pet/projects/pointrend, pet/projects/fairmot, pet/tasks/contrast, pet/projects/upsnet, pet/projects/rotatedrpn, pet/projects/crowdcounting.
- Support precise_bn for accurate sync batch normalization.
- New visualizer and measure.
- New c++/cuda implementations of box_voting and soft-nms for speeding up.
- New instance and rcnn parsing data stream.
- Support panoptic task.
- Add cpu/cuda implementations of ROIAlignRotated, box_iou_rotated and ploy_nms.
- Support nearest interpolation for ROIAlign/ROIAlignRotated.
- Add MaskIOULoss, BoundedIoULoss, DistributionFocalLoss and QualityFocalLoss.


### v0.3 (10/04/2020)

**New Features (0.3b)**
- Fix some bugs.
- New logger system.
- Speed up post-process of pet/rcnn.
- Add test-time augmentation of Grid R-CNN and Hier R-CNN.
- Add pixel_score to improve mask prediction performance.

**Improvements (0.3a)**
- Support Pytorch 1.1 to 1.4.
- Code style standardization.

**New Features (0.3a)**
- Refactoring pet/ssd implementation, mort Pet-style.
- Add pet/project/higherhrnet (TODO: accuracy bug).
- Add PFPN, NasFPN, BiFPN, HTC, Grid R-CNN, Mask Scoring R-CNN, SemSeg, ATSS, EmbedMask, RP R-CNN, Hier R-CNN, SOLO, and CBNet supports to pet/rcnn.
- Add RefineDet support to pet/ssd.
- Add EfficientNet, DLA, MobileNetV3, SPNasNet, ShuffleNetV2Plus and VoVNet to pet/models.
- New test process of pet/rcnn and pet/instance.
- New evaluation interface of pet/rcnn and pet/instance.
- Unified test augmentation implementations of pet/rcnn.
- More datasets supports, including LVIS, LIP, VIP, ATR, PASCAL-Person-Part.
- Add RepeatFactorTrainingSampler for training imbalanced datasets (LVIS and OID).
- A tool (tools/rcnn/scripts/compute_fcos_flops.py) to compute flops of FCOS like models.


### v0.2 (18/08/2019)
- Support Pytorch 1.1
- Support pet/face.
- Add project interface.
- Refactoring pet/ssd implementation.
- Refactoring pet/instance implementation, supporting keypoints, parsing, densepose uv and mask.
- pet/rcnn supports Cascade R-CNN.


### v0.1 (18/05/2019)
- First release version of Pet.
- Support Pytorch 1.0.
- New interfaces of vision tasks.
