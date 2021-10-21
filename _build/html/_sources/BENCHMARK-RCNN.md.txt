## Benchmark R-CNN



## Results and Models

|                          | Pet-v0.4 | Pet-v0.3b| detectron2 | Server |   Time   | Status |
| :----------------------: | :------: | :------: | :--------: | :----: |:--------:|:------:|
|R-50-C4-1x-ms(impr)       | 35.1     |          | 35.7       | 172    | 20200519 |  fail  |
|R-50-C5-2FC-1x-ms(impr)   | 34.9     |          | 37.3       | 210    | 20200519 |  fail  |
|R-50-FPN-1x-ms(impr)      | 37.7     |          | 37.9       | 9      | 20200519 |  check |
|R-50-FPN-3x-ms(impr)      | 40.0     |          | 40.2       | 9      | 20200512 |  check |
|R-50-FPN-1x-ms(impr)      | 38.5/35.2|          | 38.6/35.2  | 70     | 20200512 |  pass  |
|R-50-FPN-3x-ms(impr)      | 41.0/37.1|          | 41.0/37.2  | 70     | 20200512 |  pass  |
|R-101-FPN-1x-ms(impr)     | 40.7/36.9|          |            | 70     | 20200520 |  check |
|A-50d-FPN-1x-ms(impr)     | 41.5/37.4|          |            | 70     | 20200520 |  pass  |
|R-50-FPN-DCN-1x-ms(impr)  | 41.3/37.3|          | 41.5/37.5  | 70     | 20200519 |  check |
|Cascade-R-50-FPN-1x-ms(impr) | 41.6/36.6|       | 42.1/36.4  | 172    | 20200517 |  fail  |
|RS-50-FPN-1x-ms(impr)     | 40.4     |          |            | 172    | 20200512 |  check |
|RS-50-FPN-DCN-1x-ms(impr) | 42.6     |          |            | 172    | 20200512 |  check |
|Cascade-RS-50-FPN-1x-ms(impr)| 42.9     |       |            | 172    | 20200512 |  check |
|GX-3.2GF-FPN-1x           | 37.5     |          |            | 172    | 20200512 |  pass  |
|GY-32GF-FPN-1x            | 42.3     |          |            | 172    | 20200512 |  pass  |
|R-50-FPN-1x-ms(impr)      | 39.6     |          |            | 172    | 20200519 |  check |



### RPN

|                              | Pet-v0.4 | Pet-v0.3b| detectron2 | Server |   Time   | Status |
| :--------------------------: | :------: | :------: | :--------: | :----: |:--------:|:------:|
|SOLO-50-FPN-1x                | /33.0    | /32.9    |            | 20     | 20200516 |  pass  |
|FCOS-50-FPN-1x(shift)         | 38.8(38.5)| 38.8     |            | 40     | 20200513 |  pass  |
|ATSS-50-FPN-1x                | 39.3     | 39.2     |            | 60     | 20200518 |  pass  |
|RetinaNet-50-FPN-1x           | 36.4     | 36.5     |            | 60     | 20200518 |  pass  |
|RetinaNet-50-FPN-1x-ms(impr)  |          |          |    37.4    | 60     |          |  check  |
|EmbedMask-50-FPN-1x(shift)    | 38.2/33.5(37.8/33.2)| 38.2/33.5|            | 20     | 20200520 |  pass  |
|RPN-50-FPN-1x                 |   56.9   | 56.9     |            | 60     | 20200520 |  pass  |
|RPN-50-FPN-1x-ms(impr)        |   57.4   |          |    58.0    | 60     | 20200520 |  fail  |
|RPN-50-C4-1x-ms(impr)         |   51.5   |          |    51.6    | 20     | 20200521 |  pass  |

### Others

|                              | Pet-v0.4 | detectron2 | Server |   Time   | Status |
| :--------------------------: | :------: | :--------: | :----: |:--------:|:------:|
|Mask-50-C4-1x-ms(impr)        |          |  36.8/32.2 |   20   | 20200529 |        |
|Mask-50-DC5-1x-ms(impr)       |          |  38.3/34.2 |        |          |        |
|Mask-50-DCN-1x-ms(impr)       |          |  41.5/37.5 |        |          |        |
|Mask-50-FPN-1x-ms(impr)(lvis) |          |  23.6/24.4 |        |          |        |
|Mask-50-GN-3x-ms(impr)        |          |  42.6/38.6 |        |          |        |
|Mask-50-SyncBN-9x-ms-scratch(impr)|      |  43.6/39.3 |        |          |        |

