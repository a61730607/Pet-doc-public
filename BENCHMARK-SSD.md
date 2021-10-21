# Benchmark SSD


## Results and Models
Training/Inference time is measured on a sever with 8 GeForce GTX 1080 Ti.
Train on mscoco_2017_train set, test on mscoco_2017_val.

|   Network   | Backbone |  Input Size | Epochs | AP (0.6/0.6b) | Training Time (h) | Inference Time (ms)|
| :---------: | :------: | :---------: | :----: | :-----------: | :---------------: |:------------------:|
|  SSD 0.25x  |  VGG16   |  300 x 300  |   30   |  21.5 / 21.4  |         4.4       |   24 = 3 + 19 + 2  |
|    SSD      |  VGG16   |  300 x 300  |  120   |  25.4 / 25.3  |        17.2       |   24 = 3 + 19 + 2  |
|    SSD      |  VGG16   |  512 x 512  |  120   |  29.3         |        33.2       |   43 = 3 + 38 + 2  |
|  RefineDet  |  VGG16   |  320 x 320  |  120   |  28.8 / 28.9  |        19.4       |   25 = 3 + 21 + 1  |
|  RefineDet  |  VGG16   |  512 x 512  |  120   |  32.7         |        33.6       |   41 = 3 + 37 + 1  |
|    HSD      |  VGG16   |  320 x 320  |  160   |  34.0 / 34.4  |        48.0       |   34 = 3 + 29 + 2  |
|    HSD      |  VGG16   |  512 x 512  |  160   |  39.0         |       100.1       |   55 = 3 + 51 + 1  |
|             |          |             |        |               |                   |                    |
|    SSD      |  R50-c   |  512 x 512  |  120   |  30.6 / 30.7  |        26.0       |   36 = 3 + 32 + 1  |
|  RefineDet  |  R50-c   |  320 x 320  |  120   |  31.2         |        23.3       |   38 = 3 + 34 + 1  |
|  RefineDet  |  R50-c   |  512 x 512  |  120   |       / 36.7  |        37.6       |   47 = 3 + 42 + 2  |
|    HSD      |  R50-c   |  512 x 512  |  160   |  39.5 / 40.1  |        91.7       |   52 = 3 + 46 + 2  |
