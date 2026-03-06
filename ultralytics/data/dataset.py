# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
from copy import deepcopy
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path("./labels.cache").

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        assert task == "detect", "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        return []

    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        labels = []
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]])
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        return labels

    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples

class RandomHSV6Channel:
    """
    适配6通道图像的RandomHSV：仅对前3个通道（RGB）进行HSV增强
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        from .augment import RandomHSV
        self.aug = RandomHSV(hgain=hgain, sgain=sgain, vgain=vgain)
        
    def __call__(self, labels):
        img = labels["img"]
        if img.shape[2] == 6:
            # 分离RGB和IR
            rgb = np.ascontiguousarray(img[:, :, :3])
            ir = img[:, :, 3:]
            # 仅增强RGB
            labels["img"] = rgb
            labels = self.aug(labels)
            # 合并回6通道
            labels["img"] = np.concatenate([labels["img"], ir], axis=-1)
        else:
            labels = self.aug(labels)
        return labels

class FLAME2Dataset(BaseDataset):
    """
    适配FLAME2数据集的6通道（RGB+IR）数据加载器
    核心功能：
    1. 加载RGB+IR双模态图像并拼接为6通道张量
    2. 匹配YOLO格式标注文件
    3. 兼容6通道数据的增强/预处理逻辑
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """初始化FLAME2数据集加载器"""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data # 本质是cfg，即yaml文件
        # 6通道配置读取
        self.input_channels = data.get("input_channels", 6)
        self.rgb_dir = Path(data["path"]) / data["rgb_dir"]
        self.thermal_dir = Path(data["path"]) / data["thermal_dir"]
        self.label_dir = Path(data["path"]) / data["label_dir"]
        self.img_size = data.get("img_size", [254, 254])
        
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        assert self.input_channels == 6, "FLAME2数据集仅支持6通道(RGB+IR)输入"
        super().__init__(*args, **kwargs)
    
    def get_img_files(self, img_path):
        """
        重写父类方法：从train.txt/val.txt中读取数字索引，
        然后拼接为RGB图像完整路径
        """
        # 如果img_path是txt文件，读取其中的数字索引
        if isinstance(img_path, (str, Path)) and str(img_path).endswith('.txt'):
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"索引文件不存在: {img_path}")
            
            # 读取txt文件中的每一行（每行一个数字）
            with open(img_path, 'r') as f:
                indices = [line.strip() for line in f.readlines() if line.strip()]
            
            # 将数字转换为RGB图像路径
            rgb_files = []
            for idx in indices:
                rgb_path = self.rgb_dir / f"{idx}.jpg"
                if rgb_path.exists():
                    rgb_files.append(str(rgb_path))
                else:
                    LOGGER.warning(f"RGB图像不存在: {rgb_path}")
            
            return rgb_files
        else:
            # 如果不是txt文件，调用父类方法处理
            return super().get_img_files(img_path)

    def get_image_and_label(self, index):
        """重载：获取单样本的6通道图像+标注
           在父类的__getitem__()方法中调用"""
        
        # 核心修复：在 MixUp/Mosaic 过程中，index 可能是 label 字典而不是整数
        if isinstance(index, dict):
            # 必须深拷贝，否则 Mosaic 增强会修改 buffer 中的原始数据（如 pop('resized_shape')），导致后续 KeyError
            label = deepcopy(index)
        else:
            # 正常索引读取
            label = deepcopy(self.labels[index])
        
        # 如果 label 字典中已经有了加载好的 img，说明是增强过程中的二次调用，直接返回
        if "img" in label and label["img"] is not None:
            return label

        # 否则，按照常规流程加载图像
        rgb_file = Path(label["im_file"])
        thermal_file = self.thermal_dir / rgb_file.name
        label_file = self.label_dir / rgb_file.with_suffix(".txt").name

        # 加载图像
        rgb_img = self.load_image(rgb_file)
        thermal_img = self.load_image(thermal_file)
        img = np.concatenate([rgb_img, thermal_img], axis=-1)

        # 构建标准的 label 字典
        label.update({
            "img": img,
            "ori_shape": img.shape[:2],
            "resized_shape": img.shape[:2],
        })
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )

        # 核心：必须调用 update_labels_info，它会将 bboxes 转换成 'instances' 对象
        label = self.update_labels_info(label)

        # 为 Mosaic 数据增强维护缓存 (buffer)
        if self.max_buffer_length:
            if len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)
            self.buffer.append(deepcopy(label))

        return label

    def load_image(self, img_path):
        """加载单模态图像（RGB/IR），返回HWC格式numpy数组"""
        if not img_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        return img

    def load_label(self, label_file):
        """加载YOLO格式标注"""
        if not label_file.exists():
            return np.zeros((0, 5), dtype=np.float32)
        with open(label_file, "r") as f:
            lines = f.read().splitlines()
        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # YOLO格式：class_id x_center y_center width height
            parts = line.split()
            cls = int(parts[0])
            bbox = np.array(parts[1:5], dtype=np.float32)
            labels.append([cls, *bbox])
        return np.array(labels, dtype=np.float32).reshape(-1, 5)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        重载缓存逻辑：适配6通道数据的标签缓存
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        
        # 自定义验证逻辑：检查RGB/IR/标注文件完整性
        def verify_flame2_sample(im_file):
            rgb_file = Path(im_file)
            thermal_file = self.thermal_dir / rgb_file.name
            label_file = self.label_dir / rgb_file.with_suffix(".txt").name
            
            # 验证RGB图像
            (path, cls), nf, nc, msg = verify_image(((rgb_file, None), "rgb"))
            if not nf:
                return None, 1, 0, 0, 0, f"RGB图像损坏: {rgb_file}"
            
            # 验证IR图像
            (path, cls), nf, nc, msg = verify_image(((thermal_file, None), "thermal"))
            if not nf:
                return None, 1, 0, 0, 0, f"IR图像损坏: {thermal_file}"
            
            # 验证标注
            if label_file.exists():
                lb = self.load_label(label_file)
                ne_f = 1 if len(lb) == 0 else 0
                nc_f = 0
            else:
                lb = np.zeros((0, 5), dtype=np.float32)
                ne_f = 1
                nc_f = 0
            
            # 获取图像尺寸 (H, W)
            # 注意：由于 FLAME2Dataset.load_image 会 resize 图像到 self.img_size
            # 所以这里的 shape 应该是 self.img_size
            img_shape = (self.img_size[0], self.img_size[1])

            return {
                "im_file": str(rgb_file),
                "thermal_file": str(thermal_file),
                "cls": lb[:, 0:1],
                "bboxes": lb[:, 1:],
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
                "shape": img_shape, # 必须包含此键，否则 rect=True 时会报 KeyError: 'shape'
            }, 0, 1, ne_f, nc_f, "" # missing, found, empty, corrupt

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_flame2_sample,
                iterable=self.im_files,
            )
            pbar = TQDM(results, desc=desc, total=total)
            for res in pbar:
                if res is None:
                    continue
                label, nm_f, nf_f, ne_f, nc_f, msg = res
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if label:
                    x["labels"].append(label)
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """重载标签读取逻辑：适配6通道缓存"""
        # 确保缓存目录存在
        cache_dir = Path(self.label_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用输入路径的 stem (如 train 或 val) 作为缓存文件名，避免冲突
        prefix = Path(self.img_path).stem if self.img_path else "flame2"
        cache_path = cache_dir / f"{prefix}.cache"
        
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        # 读取缓存标签
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        
        # 核心修复：BaseDataset 期望 self.labels 是一个包含 label 字典的列表
        # 如果缓存中直接存了 label 字典，这里直接返回即可
        return labels

    def build_transforms(self, hyp=None):
        """
        重载数据增强：适配6通道图像
        核心修改：确保增强操作兼容6通道（而非默认3通道）
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # 自定义6通道增强逻辑
            transforms = self._build_6channel_transforms(self, self.imgsz, hyp)
        else:
            # 验证阶段仅做尺寸调整
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        
        # 格式转换：适配6通道输出
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio if hyp else 4,
                mask_overlap=hyp.overlap_mask if hyp else 0.5,
                bgr=hyp.bgr if (hyp and self.augment) else 0.0,
            )
        )
        return transforms

    def _build_6channel_transforms(self, dataset, imgsz, hyp):
        """
        构建6通道数据增强流水线
        核心：所有增强操作同时作用于RGB和IR通道（保持模态一致性）
        """
        from .augment import (
            RandomFlip,
            RandomPerspective,
            MixUp,
            Mosaic,
        )
        transforms = Compose([
            Mosaic(dataset, p=hyp.mosaic, n=4, imgsz=imgsz),
            RandomHSV6Channel(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="horizontal", p=hyp.flipud),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                border=(-imgsz // 2, -imgsz // 2),
            ),
            MixUp(dataset, p=hyp.mixup),
        ])
        return transforms

    def update_labels_info(self, label):
        """重载标签格式：适配6通道数据"""
        # 备份这些键，防止 pop 之后在 Mosaic 增强中找不到
        label["bboxes"] = label.get("bboxes", np.zeros((0, 4), dtype=np.float32))
        label["segments"] = label.get("segments", [])
        label["keypoints"] = label.get("keypoints", None)
        label["bbox_format"] = label.get("bbox_format", "xywh")
        label["normalized"] = label.get("normalized", True)

        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        
        # 恢复这些键，以便 Mosaic 增强后续步骤（如 _mosaic4）使用
        label["bboxes"] = bboxes
        label["segments"] = segments
        label["keypoints"] = keypoints
        label["bbox_format"] = bbox_format
        label["normalized"] = normalized
        
        return label

    @staticmethod
    def collate_fn(batch):
        """
        重载批次拼接：适配6通道图像张量
        """
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                new_batch[k] = torch.stack(
                    [torch.from_numpy(v).permute(2, 0, 1) if isinstance(v, np.ndarray) else v for v in value]
                )
            elif k == "instances":
                new_batch[k] = [v for v in value]
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                if all(v is None for v in value):
                    new_batch[k] = None
                else:
                    tensors = [
                        torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in value if v is not None
                    ]
                    new_batch[k] = torch.cat(tensors, 0)
            else:
                new_batch[k] = value
        
        if "batch_idx" in new_batch:
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # add target image index for build_targets()
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
            
        return new_batch
