### Benchmark Face


1. **Casia**:
   Train set: Identities(10K), images(0.5M).


   |   Backbone   |   Head  |   Loss  |  Flops/Params  |  lfw  |  agedb30  |  cfp_fp  |
   | :----------: | :-----: | :-----: | :------------: | :---: | :-------: | :------: |
   | MobileFaceNet| ArcFace | Softmax | 233.8M/1M      | 98.95 |   92.12   |  93.43   |
   | MobileNetV3  | ArcFace | Softmax | 210.86M/3.2M   | 98.98 |   92.22   |  94.83   |
   | L-ResNet-50  | ArcFace | Softmax | 4087.14M/74.9M | 99.38 |   93.20   |  95.61   |
   | ResNetIR-50  | ArcFace | Softmax | 6296.49M/43.6M | 99.43 |   94.48   |  96.79   |
   | ResNetIRSE-50| ArcFace | Softmax | 6296.71M/43.8M | 99.40 |   94.00   |  97.07   |


2. **MS-1M-Retinaface**:
   Train set: classes(85K), images(5.8M).


   |   Backbone   |   Head  |   Loss  |  Flops/Params  |  lfw  |  agedb30  |  cfp_fp  |
   | :----------: | :-----: | :-----: | :------------: | :---: | :-------: | :------: |
   | MobileFaceNet| ArcFace | Softmax | 233.8M/1M      |       |           |          |
   | MobileNetV3  | ArcFace | Softmax | 210.86M/3.2M   |       |           |          |
   | L-ResNet-50  | ArcFace | Softmax | 4087.14M/74.9M |       |           |          |
   | ResNetIR-50  | ArcFace | Softmax | 6296.49M/43.6M |       |           |          |
   | ResNetIRSE-50| ArcFace | Softmax | 6296.71M/43.8M |       |           |          |
