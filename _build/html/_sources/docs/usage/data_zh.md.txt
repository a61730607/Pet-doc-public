# 数据加载教程

## 介绍

Pet提供了一整套详细的数据读取流程，涵盖多个读取模块。通过
`tools/{type}/[project/subtask]/train_net.py`获取脚本信息，指定`tools/train_net_all.py`或`tools/test_net_all.py`实现训练或测试阶段的数据读取。下面将从通用训练脚本`tools/train_net_all.py`切入讲解数据读取流程，进而以`tools/vision/train_net.py`作为窗口介绍整套读取流程，主要涵盖数据制备与数据加载。

* **数据制备**:对于不同的视觉任务，Pet支持在多种数据集上进行模型的训练和测试，并且规定了Pet标准的数据集源文件的文件结构与标注的格式。
* **数据加载**:Pet实现了一套标准的数据载入接口，同时提供了多种在线数据增强方式如尺度变换、旋转、翻折等使得神经网络训练具有更好的泛化效果。

## 数据加载流程

通过`tools/train_net_all.py`获取指定的配置信息。根据配置信息创建指定数据集类、数据集加载器类、数据训练周期。代码如下：
```python
# Create training dataset and loader
dataset = build_dataset(cfg, is_train=True) #创建指定数据集类，读取数据
start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter) # 创建指数据集加载器类，对读取的数据集进行迭代加载
max_iter = len(train_loader) 
iter_per_epoch = max_iter // cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS 
```
* **创建指定数据集类，读取数据**
* **创建指数据集加载器类，对读取的数据集进行迭代加载**
* **根据任务要求提供多种数据增强方式**
* **根据任务要求提供评估核验处理**
## 训练流程

每个具体数据读取过程都包含训练阶段(train)和测试阶段（test），此处只介绍训练逻辑，具体函数应用细节请看API文档。

两个阶段的数据读取流程类似，本教程将以训练阶段(train)为切入口详细介绍数据的读取过程。
### (1) 确定数据集及训练任务
Pet支持的全部数据读取方式如下代码所述，其中`ImageFolderDataset`为自定义数据集提供一个更简单的方法，即直接调用ImageFolder，它是torchvision.datasets里的函数。
```python
DATASET_TYPES = {
    "cifar_dataset": CifarDataset,
    "coco_dataset": COCODataset,
    "coco_instance_dataset": COCOInstanceDataset,
    "image_folder_dataset": ImageFolderDataset,
}
```
Pet利用PyTorch提供的数据处理类`ImageFolder`作为数据加载器，ImageFolder将父文件夹下的子文件夹读取创建字典，并对应为类别标签，文件夹目录在YAML文件下DATASET.DATA_ROOT中进行设置。为了达到数据增强的效果，以减少一定的过拟合现象，我们使用PyTorch自带的transforms函数对数据进行预处理，使用水平方向上以一定的概率进行图片翻转来进行数据增强，将图片进行随机裁剪至224*224送入网络。除此之外，我们还支持另一种数据增强方式——mixup。cfg.TRAIN.AUG则是可以在配置文件中针对数据增强可以被设置的参数。ImageFolder类的具体实现请阅读PyTorch官方代码

Pet根据main函数配置文件确定训练任务。Pet提供的训练任务如下：
```python
    ann_types = set()
    if cfg.MODEL.HAS_BOX:
        ann_types.add('bbox')
    if cfg.MODEL.HAS_MASK:
        ann_types.add('mask')
    if cfg.MODEL.HAS_SEMSEG:
        ann_types.add('semseg')
    if cfg.MODEL.HAS_PANOSEG:
        ann_types.add('panoseg')
    if cfg.MODEL.HAS_KEYPOINTS:
        ann_types.add('keypoints')
    if cfg.MODEL.HAS_PARSING:
        ann_types.add('parsing')
    if cfg.MODEL.HAS_UV:
        ann_types.add('uv')
        raise NotImplementedError
    if cfg.MODEL.HAS_CLS:
        ann_types.add('cls')
```
### (2) 根据数据集索引读取图像
```
        dataset = dataset_obj(root, ann_file, ann_types, ann_fields,
                              transforms=transforms,
                              is_train=is_train,
                              filter_invalid_ann=is_train,
                              filter_empty_ann=is_train,
                              filter_crowd_ann=True,    # TO Check, TODO: filter ignore
                              bbox_file=bbox_file,
                              image_thresh=image_thresh,
                              mosaic_prob=mosaic_prob)
        datasets.append(dataset)
```
### (3) 将所有数据集连接成一个数据集
```Python
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        logging_rank(f"Concatenate datasets: {dataset_list}.")
        # 设置日志和加载配置选项
    else:
        dataset = datasets[0]

    return dataset
```
### (4) 设置批训练加载器（dataloader）
加载器功能如下：
```Python
    data_loader = DataLoader(
        datasets,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
```

* **重复因子训练采样器**：设定每轮图像的重复数量，代码如下
```Python
def make_data_sampler(cfg, dataset, shuffle):
    if cfg.DATA.SAMPLER.TYPE == "RepeatFactorTrainingSampler":
        return RepeatFactorTrainingSampler(dataset, cfg.DATA.SAMPLER.RFTSAMPLER, shuffle=shuffle)
    else:
        return DistributedSampler(dataset, shuffle=shuffle)
```
* **批量参数采样器**：试图寻求最小批次，代码如下
```Python
def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, drop_last=False):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        group_ids = GroupedBatchSampler.get_group_ids(dataset, aspect_grouping)
        batch_sampler = GroupedBatchSampler(group_ids, sampler, images_per_batch, drop_last)
    else:
        batch_sampler = BatchSampler(sampler, images_per_batch, drop_last)
    return batch_sampler
```
通过采样器，产生一个新的批次区间。它规定了同一组的元素应该出现在`batch_size`中的数量。并寻求一个符合实际训练需求的小批次空间。

* **分布式采样器**：用于限制数据加载到分布式子集的采样器，并进行数据增强。代码如下：
```Python
if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(int(self.repeat_time))]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
```
### (5) 数据集变换模块（transform）

在Pet教程中，该模块主要负责提供多种数据增强方式：
* **归一化**：
```Python
def normalize(cfg, is_train):
    return T.Normalize(
        mean=cfg.DATA.PIXEL_MEAN,
        std=cfg.DATA.PIXEL_STD,
        mode=cfg.DATA.FORMAT.IMAGE,
    )
```
* **随机水平翻转**：
```Python
def random_horizontal_flip(cfg, is_train):
    return T.RandomFlip(method='horizontal', prob=0.5)
```
* **随机垂直翻转**：
```Python
def random_vertical_flip(cfg, is_train):
    return T.RandomFlip(method='vertical', prob=0.5)
```
* **色彩抖动**：
```Python
def color_jitter(cfg, is_train):
    if is_train:
        return T.ColorJitter(
            brightness=cfg.TRAIN.COLOR_JITTER.BRIGHTNESS,
            contrast=cfg.TRAIN.COLOR_JITTER.CONTRAST,
            saturation=cfg.TRAIN.COLOR_JITTER.SATURATION,
            hue=cfg.TRAIN.COLOR_JITTER.HUE,
        )
    else:
        raise NotImplementedError
```
* **调整尺寸**：
```Python
def resize(cfg, is_train):
    if is_train:
        return T.Resize(
            min_size=cfg.TRAIN.RESIZE.SCALES,
            max_size=cfg.TRAIN.RESIZE.MAX_SIZE,
            scales_sampling=cfg.TRAIN.RESIZE.SCALES_SAMPLING,
            scale_factor=cfg.TRAIN.RESIZE.SCALE_FACTOR,
        )
    else:
        raise NotImplementedError
```
* **随机裁剪**：
```Python
def random_crop(cfg, is_train):
    if is_train:
        return T.RandomCrop(
            crop_sizes=cfg.TRAIN.RANDOM_CROP.CROP_SCALES,
            iou_threshs=cfg.TRAIN.RANDOM_CROP.IOU_THS,
            border=cfg.TRAIN.RANDOM_CROP.BORDER,
            cat_max_ths=cfg.TRAIN.RANDOM_CROP.CAT_MAX_THS,
            ignore_label=cfg.TRAIN.RANDOM_CROP.IGNORE_LABEL,
            pad_pixel=cfg.DATA.PIXEL_MEAN,
            mode=cfg.DATA.FORMAT.IMAGE,
        )
    else:
        raise NotImplementedError
```
* **中心裁剪**：
```Python
def center_crop(cfg, is_train):
    config = eval(f"cfg.{'TRAIN' if is_train else 'TEST'}")
    return T.CenterCrop(
        crop_sizes=config.CENTER_CROP.CROP_SCALES,
    )
```
* **SSD中的裁剪和扩张**：
```Python
def ssd_crop_and_expand(cfg, is_train):
    if is_train:
        return T.SSDCropAndExpand(
            expand_prob=cfg.TRAIN.CROP_AND_EXPAND.PROB,
            pad_pixel=cfg.DATA.PIXEL_MEAN,
            mode=cfg.DATA.FORMAT.IMAGE,
        )
    else:
        raise NotImplementedError
```
* **检测任务中高效的裁剪和尺寸调整**：
```Python
def efficient_det_resize_crop(cfg, is_train):
    if is_train:
        return T.EfficientDetResizeCrop(
            cfg.TRAIN.EFFICIENT_DET_RESIZE_CROP.SIZE,
            cfg.TRAIN.EFFICIENT_DET_RESIZE_CROP.SCALE,
        )
    else:
        return T.EfficientDetResizeCrop(
            cfg.TEST.EFFICIENT_DET_RESIZE_CROP.SIZE,
            (1, 1),
        )
```
* **随机尺寸调整并裁剪图像**：
```Python
def random_resized_crop(cfg, is_train):
    if is_train:
        return T.RandomResizedCrop(
            size=cfg.TRAIN.RANDOM_RESIZED_CROP.SIZE,
            scale=cfg.TRAIN.RANDOM_RESIZED_CROP.SCALE,
            ratio=cfg.TRAIN.RANDOM_RESIZED_CROP.RATIO,
            interpolation=cfg.TRAIN.RANDOM_RESIZED_CROP.INTERPOLATION,
        )
    else:
        raise NotImplementedError
```
* **自动调整图像**：
```Python
def auto_aug(cfg, is_train):
    class AutoAug(T.Transform):
        __ann_types__ = {'cls'}
        __image_types__ = {Image.Image}

        def __init__(self, tran):
            self.tran = tran

        def __call__(self, image, target):
            return self.tran(image), target

    if is_train:
        config: str = cfg.TRAIN.AUTO_AUG.CONFIG
        aa_params = {
            "translate_const": int(cfg.TRAIN.AUTO_AUG.CROP_SIZE * 0.45),
            "img_mean": T.Transform.convert_pixel_format(
                cfg.DATA.PIXEL_MEAN, from_mode=cfg.DATA.FORMAT.IMAGE, to_mode="rgb255"),
            "interpolation": T.INTERPOLATION_METHOD[cfg.TRAIN.AUTO_AUG.INTERPOLATION],
        }
        if config.startswith('rand'):
            tran = T.rand_augment_transform(config, aa_params)
        elif config.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            tran = T.augment_and_mix_transform(config, aa_params)
        else:
            tran = T.auto_augment_transform(config, aa_params)
        return AutoAug(tran)
    else:
        raise NotImplementedError
```
### (6) 后置模块（post_process）

在Pet教程中，该模块主要负责模型评估、核验等后置处理：
* **R-CNN后置处理**：
```
class CNNPostProcessor(object):
    def __init__(self, cfg, datasets):
        """ R-CNN postprocessing

        Args:
            cfg (CfgNode)
        """
        self.cfg = cfg
        self.datasets = datasets

        self.visualizer = None if not self.cfg.VIS.ENABLED else Visualizer(cfg.VIS, datasets)
        self.evaluator = None if not self.cfg.EVAL.ENABLED else Evaluator(cfg, datasets, dataset_catalog)
```
采用R-CNN算法实现各项任务前，通过各项任务的实现功能来准备评估结果并做好各项任务的准备工作，函数、评价指标等。 
* **分类任务的后置处理**：
```Python
def prepare_cls_results(cfg, results, image_ids, dataset, targets=None):
    ims_labels = []
    cls_results = []
    for i, result in enumerate(results):
        scores = result['cls_scores'].tolist()
        labels = result['label'].tensor.tolist()
        ims_labels.append(labels)
        image_id = image_ids[i]
        cls_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": label,
                    "score": score,
                }
                for label, score in zip(labels, scores)
            ]
        )
    return cls_results, ims_labels
```
* **检测任务的后置处理**：
```Python
def prepare_box_results(cfg, results, image_ids, dataset, targets=None):
    box_results = []
    ims_dets = []
    ims_labels = []
    if cfg.MODEL.RPN.RPN_ONLY:
        for i, result in enumerate(results):
            image_id = image_ids[i]
            result.add_extra_data('image_id', image_id)
        return results, None, None

    for i, result in enumerate(results):
        if len(result) == 0:
            ims_dets.append(None)
            ims_labels.append(None)
            continue
        image_id = image_ids[i]
        original_id = dataset.ids[image_id]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        bbox_struct = result['bbox']
        if cfg.MODEL.ROI_HEAD.GRID_ON:
            bbox_struct.tensor = get_grid_results(cfg, result["grid"], result['bbox'])
        bbox_struct = bbox_struct.resize((image_width, image_height))
        boxes, box_mode = bbox_struct.tensor, bbox_struct.mode
        scores = result['bbox_scores']
        labels = result['label'].tensor
        ims_dets.append(np.hstack((boxes.cpu(), scores.cpu()[:, np.newaxis])).astype(np.float32, copy=False))
        boxes = BoxMode.convert(boxes, box_mode, BoxMode.XYWH).tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        ims_labels.append(labels)
        mapped_labels = [dataset.contiguous_id_to_category_id[i] for i in labels]
        box_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return box_results, ims_dets, ims_labels
```
* **图像解析任务的后置处理**：
```Python
if cfg.MODEL.GLOBAL_HEAD.INSSEG.EMBED_MASK_ON:
            rles, mask_pixel_scores = get_embedmask_results(
                cfg, masks, result, input_w, input_h, image_height, image_width
            )
            # calculating quality scores
            mask_bbox_scores = result["mask_scores"]  # mask_bbox_scores
            mask_iou_scores = torch.ones(
                mask_bbox_scores.size()[0], dtype=torch.float32, device=mask_bbox_scores.device
            )
            alpha, beta, gamma = cfg.MODEL.EMBED_MASK.QUALITY_WEIGHTS
            _dot = (torch.pow(mask_bbox_scores, alpha)
                    * torch.pow(mask_iou_scores, beta)
                    * torch.pow(mask_pixel_scores, gamma))
            scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        elif cfg.MODEL.GLOBAL_HEAD.INSSEG.POLAR_MASK_ON:
            rles, mask_pixel_scores = get_polarmask_results(
                cfg, masks, result, input_w, input_h, image_height, image_width
            )
            # calculating quality scores
            mask_bbox_scores = result["mask_scores"]  # mask_bbox_scores
            mask_iou_scores = torch.ones(
                mask_bbox_scores.size()[0], dtype=torch.float32, device=mask_bbox_scores.device
            )
            alpha, beta, gamma = cfg.MODEL.POLAR_MASK.QUALITY_WEIGHTS
            _dot = (torch.pow(mask_bbox_scores, alpha)
                    * torch.pow(mask_iou_scores, beta)
                    * torch.pow(mask_pixel_scores, gamma))
            scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
```
* **关键点任务的后置处理**：
```Python
keypoints[:, :, :2] -= cfg.MODEL.KEYPOINT.OFFSET
        alpha, beta, gamma = cfg.MODEL.KEYPOINT.QUALITY_WEIGHTS
        _dot = (torch.pow(kpt_bbox_scores, alpha)
                * torch.pow(kpt_iou_scores, beta)
                * torch.pow(kpt_pixel_scores, gamma))
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        mapped_labels = [dataset.contiguous_id_to_category_id[i] for i in labels]
        kpt_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "keypoints": keypoint.flatten().tolist(),
                    "score": scores[k]
                }
                for k, keypoint in enumerate(keypoints)
            ]
```
* **parsing任务的后置处理**：
```Python
def prepare_parsing_results(cfg, results, image_ids, dataset, targets=None):
    pars_results = []
    ims_parss = []
    output_folder = os.path.join(cfg.MISC.CKPT, 'test')
    _, output_semseg_prob = get_semseg_params(cfg)
    for i, result in enumerate(results):
        image_id = image_ids[i]
        original_id = dataset.ids[image_id]
        if len(result) == 0:
            ims_parss.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        result = result.resize((image_width, image_height))
        semseg = result["semseg"].squeeze(0).cpu().numpy().transpose((1, 2, 0)) \
            if cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON and output_semseg_prob else None
        parsings = result["parsing_prob"]
        parsings, parsing_instance_pixel_scores, parsing_part_pixel_scores = get_parsing_results(cfg, parsings,
                                                                                                 result['bbox'])
        # calculating quality scores
        parsing_bbox_scores = result["bbox_scores"]
        parsing_iou_scores = result["parsing_iou_scores"]
        alpha, beta, gamma = cfg.MODEL.PARSING.QUALITY_WEIGHTS
        instance_dot = (torch.pow(parsing_bbox_scores, alpha)
                        * torch.pow(parsing_iou_scores, beta)
                        * torch.pow(parsing_instance_pixel_scores, gamma))
        instance_scores = torch.pow(instance_dot, 1. / sum((alpha, beta, gamma))).tolist()
        part_dot = torch.stack([torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta)] *  # noqa: W504
                               (cfg.MODEL.PARSING.NUM_PARSING - 1), dim=1) * torch.pow(parsing_part_pixel_scores, gamma)
        part_scores = torch.pow(part_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = result['label'].tensor.tolist()
        ims_parss.append(parsings)
        mapped_labels = [dataset.contiguous_id_to_category_id[i] for i in labels]
        parsings, instance_scores = generate_parsing_result(
            parsings, instance_scores, part_scores, parsing_bbox_scores.tolist(), semseg=semseg, img_info=img_info,
            output_folder=output_folder, score_thresh=cfg.MODEL.PARSING.SCORE_THRESH,
            semseg_thresh=cfg.MODEL.PARSING.SEMSEG_SCORE_THRESH, parsing_nms_thres=cfg.MODEL.PARSING.PARSING_NMS_TH,
            num_parsing=cfg.MODEL.PARSING.NUM_PARSING
        )
        pars_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "parsing": csr_matrix(parsing),
                    "score": instance_scores[k]
                }
                for k, parsing in enumerate(parsings)
            ]
        )
    ims_parss = ims_parss if len(ims_parss) else [None]
    return pars_results, ims_parss
```
* **semseg任务的后置处理**：
```Python
def prepare_semseg_results(cfg, results, image_ids, dataset, targets=None):
    semseg_results = []
    output_folder = os.path.join(cfg.MISC.CKPT, 'test')
    for i, result in enumerate(results):
        if 'semseg' not in result:
            return [], semseg_results
        image_id = image_ids[i]
        img_info = dataset.get_img_info(image_id)
        semseg_preds = result['semseg'].squeeze(0).argmax(dim=0)
        semseg_preds = semseg_preds.cpu().numpy()
        semseg_results.append(semseg_preds)
        semseg_png(semseg_preds, dataset, img_info, output_folder)
    return [], semseg_results
```
* **panoptic任务的后置处理**：
```Python
def prepare_panoptic_results(cfg, results, image_ids, dataset, targets=None):
    import io
    from PIL import Image
    panos_results = []
    ims_panos = []
    for i, result in enumerate(results):
        image_id = image_ids[i]
        original_id = dataset.ids[image_id]
        img_info = dataset.get_img_info(image_id)
        image_height = img_info["height"]
        image_width = img_info["width"]
        # input_w, input_h = result.size
        contiguous_id_to_pano_id = dataset.contiguous_id_to_pano_id
        segments_info, ims_pano = get_pfpnnet_results(
            cfg, result, image_height, image_width, contiguous_id_to_pano_id
        )
        ims_panos.append(ims_pano)
        with io.BytesIO() as out:
            if not isinstance(ims_pano, np.ndarray):
                if ims_pano.is_cuda:
                    ims_pano = ims_pano.cpu()
                ims_pano = ims_pano.numpy()
            Image.fromarray(id2rgb(ims_pano)).save(out, format="PNG")
            panos_results.extend(
                [
                    {
                        "image_id": original_id,
                        "file_name": img_info["file_name"].replace('jpg', 'png'),
                        "png_data": out.getvalue(),
                        "segments_info": segments_info
                    }
                ]
            )

    return panos_results, ims_panos
```
### (7) iter训练周期
Pet以**iter**为周期的训练模式，Pet还有以**epoch**为周期的训练模式，其代码设置如下：
```python
iter_per_epoch = len(train_loader)
...
max_iter = iter_per_epoch
```
两种训练模式的区别：
* **iter**：整个iter训练过程中，每个训练样本只加载一次，即完成训练。
* **epoch**：所有样本数据加载完一次视为一个epoch，即有多少个epoch，每个样本在整个训练过程中就被加载过几次。

对应的两种模式的训练函数实现逻辑也会有所不同。

通过介绍Pet训练阶段数据读取流程可以发现，Pet的整个数据读取工作流遵循“任务+算法+对象”的思想，因此，在使用Pet读取数据的过程中，用户可以通过自定义算法，实现多样化的数据流。
