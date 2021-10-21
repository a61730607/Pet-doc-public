### Benchmark SemSeg

Description:
**mIoU/mAcc** stands for mean IoU, mean accuracy of each class and all pixel accuracy respectively.
Training time is measured on a sever with 8 GeForce GTX 1080 Ti.

1. **ADE20K**:
   Train Parameters: classes(150), epochs(120).
   Test Parameters: classes(150).

   - Setting: train on **train** (20210 images) set and test on **val** (2000 images) set.

   |  Network   |  Backbone   |  mIoU/mAcc  | Training Time |
   | :--------: | :---------: | :---------: | :-----------: |
   |  PSPNet    |    R50c     | 42.36/79.83 |      35h      |
   |  PSPNet    |    R101c    | 43.65/81.02 |               |
   | DeepLabv3  |    R50c     | 43.01/80.65 |      34h      |
   | DeepLabv3  |    R101c    |             |               |
   | DeepLabv3+ |    R50c     |             |               |
   | DeepLabv3+ |    R101c    |             |               |
   |  UperNet   |    R50c     | 42.78/80.40 |      36h      |
   |  UperNet   |    R101c    |             |               |
   | SemSegFPN  |    R50c     | 39.67/78.88 |               |
   | SemSegFPN  |    R101c    |             |               |
   |   HRNet    |  HRNet-W48  | 44.18/81.21 |               |
   |    OCR     |  HRNet-W48  | 45.82/82.01 |               |


2. **Cityscapes**:
   Train Parameters: classes(19), epochs(200).
   Test Parameters: classes(19).

   - Setting: train on **fine_train** (2975 images) set and test on **fine_val** (500 images) set.

   |  Network   |  Backbone   |  mIoU/mAcc  | Training Time |
   | :--------: | :---------: | :---------: | :-----------: |
   |  PSPNet    |    R50c     | 77.54/95.87 |               |
   |  PSPNet    |    R101c    |             |               |
   | DeepLabv3  |    R50c     | 78.83/96.05 |               |
   | DeepLabv3  |    R101c    |             |               |
   | DeepLabv3+ |    R50c     |             |               |
   | DeepLabv3+ |    R101c    |             |               |
   |  UperNet   |    R50c     | 78.35/96.01 |               |
   |  UperNet   |    R101c    |             |               |
   | SemSegFPN  |    R50c     |             |               |
   | SemSegFPN  |    R101c    |             |               |
   |   HRNet    |  HRNet-W48  |             |               |
   |    OCR     |  HRNet-W48  |             |               |


3. **PSACAL VOC 2012**:
   Train Parameters: classes(21), epochs(60).
   Test Parameters: classes(21).

   - Setting: train on **train_aug** (10582 images) set and test on **val** (1449 images) set.

   |  Network   |  Backbone   |  mIoU/mAcc  | Training Time |
   | :--------: | :---------: | :---------: | :-----------: |
   |  PSPNet    |    R50c     | 78.75/94.92 |               |
   |  PSPNet    |    R101c    |             |               |
   | DeepLabv3  |    R50c     | 79.00/94.96 |               |
   | DeepLabv3  |    R101c    |             |               |
   | DeepLabv3+ |    R50c     |             |               |
   | DeepLabv3+ |    R101c    |             |               |
   |  UperNet   |    R50c     | 78.12/94.78 |               |
   |  UperNet   |    R101c    |             |               |
   | SemSegFPN  |    R50c     | 71.56/93.14 |               |
   | SemSegFPN  |    R101c    |             |               |
   |   HRNet    |  HRNet-W48  |             |               |
   |    OCR     |  HRNet-W48  |             |               |


4. **CIHP**:
   Train Parameters: classes(20), epochs(140).
   Test Parameters: classes(20).

   - Setting: train on **fine_train** (28280 images) set and test on **fine_val** (5000 images) set.

   |  Network   |  Backbone   |  mIoU/mAcc  | Training Time |
   | :--------: | :---------: | :---------: | :-----------: |
   |  PSPNet    |    R50c     | 59.70/91.23 |               |
   |  PSPNet    |    R101c    |             |               |
   | DeepLabv3  |    R50c     |             |               |
   | DeepLabv3  |    R101c    |             |               |
   | DeepLabv3+ |    R50c     |             |               |
   | DeepLabv3+ |    R101c    |             |               |
   |  UperNet   |    R50c     |             |               |
   |  UperNet   |    R101c    |             |               |
   | SemSegFPN  |    R50c     |             |               |
   | SemSegFPN  |    R101c    |             |               |
   |   HRNet    |  HRNet-W48  |             |               |
   |    OCR     |  HRNet-W48  |             |               |
