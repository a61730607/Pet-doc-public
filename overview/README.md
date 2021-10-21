<p align="center"><img width="30%" src="docs/logo.png" /></p>

--------------------------------------------------------------------------------

# Pet 0.7a
Pytorch Efficient Toolbox (Pet) for Computer Vision.

<p align="center"><img width="90%" src="docs/vis_show.png" /></p>

## Introduction
Pet is an open source supporting many tasks in computer vision based on pytorch.The toolbox can support following tasks.
- [x] **Image Classification** (covnet..., transformer...)
- [x] **Object Detection** (rpn/ga_rpn, ssd/refinedet/hsd, retinanet/fcos/atss/gfl(v1/v2)/ota/centernet(v1/v2)/onenet, /faster/cascade/grid r-cnn, ...)
- [x] **Instance Segmentation** (mask/mask-scoring r-cnn, pointrend, solo(v1/v2)/condinst/embed_mask/polar_mask, ...)
- [x] **Semantic Segmentation** (pspnet/deeplabv3/upernet/semsegfpn/ocrnet/segformer/maskformer, ...)
- [x] **Panoptic Segmentation** (pfpn/maskformer, ...)
- [x] **Pose Estimation** (keypoints r-cnn, simple-baseline/hrnet, ...)
- [x] **Human Parsing** (parsing/rp r-cnn, qanet/pdrnet, ...)

Pet also supports some other issues.
- [x] **Long-tailed recognition** (rfs/eql(v1/v2)/seasaw-loss/disalign/bags, decoupling, ...)
- [x] **Self-supervised Learning** (moco(v1/v2/v3), infomin, dino, ...)

**For different tasks, training/testing scripts and more content are under the corresponding path**

### Major Features

- **Functions**

  Excel at various tasks of Computer vision.

  Provide implementations of latest deep learning algorithms.

  Aim to help developer start own research quickly.

- **Trait**

  Modularization, flexible configuration.

  Implementation of state-of-the-art algorithm in Computer Vision.

  Clear and easy-learning process of training & inference.

- **Contrast**

  Support various kinds of tasks in Computer Vision.

  Provide numerous high-quality pre-training model.

  Own unique advantages in speed and accuracy.

- **Expand**

  Easy to validate new ideas using provided various of basic function.

  Code with uniform format and style which is easy to expand.

  Update and expand constantly.  Custom extension is supported.


## Updates

 Please check the [`CHANGELOG`](..//CHANGELOG.md)


## Installation

 Please find detailed installation instructions for Pet in [`INSTALL`](../INSTALL.md).


## Getting started

 Please find detailed tutorial for getting started in [`GETTING_STARTED`](../GETTING_STARTED.md).

 **Quick training**

 ```
# 8 GPUs
python tools/train_net_all.py --cfg cfgs/vision/COCO/e2e_faster-impr_rcnn_R-50-FPN_1x_ms.yaml

# Specify GPUs
python tools/train_net_all.py --cfg cfgs/vision/COCO/e2e_faster-impr_rcnn_R-50-FPN_1x_ms.yaml --gpu_id 0,1,2,3
```

 **Quick testing**
  ```
# 8 GPUs
python tools/test_net_all.py --cfg ckpts/vision/COCO/e2e_faster-impr_rcnn_R-50-FPN_1x_ms/e2e_faster-impr_rcnn_R-50-FPN_1x_ms.yaml

# Specify GPU
python tools/test_net_all.py --cfg ckpts/vision/COCO/e2e_faster-impr_rcnn_R-50-FPN_1x_ms/e2e_faster-impr_rcnn_R-50-FPN_1x_ms.yaml --gpu_id 0,1
```

## Benchmarks and Model Zoo

We provide a large set of baseline results and trained models available for download in [`BENCHMARK_CLS`](../BENCHMARK-CLS.md),
[`BENCHMARK_FACE`](../BENCHMARK-FACE.md), [`BENCHMARK_INSTANCE`](../BENCHMARK-INSTANCE.md), [`BENCHMARK_RCNN`](../BENCHMARK-RCNN.md),
[`BENCHMARK_SEMSEG`](../BENCHMARK-SEMSEG.md), [`BENCHMARK_SSD`](../BENCHMARK-SSD.md).

We also provide a large set of pre-trained backbone weights for network initialization (including 3rdparty implementations) in [`MODEL_ZOO`](../MODEL_ZOO.md).


## License

Pet is released under the MIT License (refer to the LICENSE file for details).


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
