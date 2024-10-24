import copy
import os
import json

import cv2
import numpy as np
from collections import defaultdict
import torch
import torch.utils.data
import torchvision
from PIL import ImageFile
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO




class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_coco_poly_to_mask_v2(segmentations, height, width):
    masks = []
    num_polygons = len(segmentations)
    # masks = torch.zeros((num_polygons, width, height))
    for i in range(num_polygons):
        polygon = segmentations[i]
        img = np.zeros((width, height), np.uint8)
        cv2.drawContours(img, [np.array(polygon)], -1, 1, -1)
        mask = torch.as_tensor(img, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
        # masks[i] = torch.tensor(img, dtype=torch.uint8)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    "This method transforms the coco annotation to the format needed for pytorch GeneralisedRCNN ie it's important"

    def __call__(self, image, target):
        w, h = image.size
        keep = None

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        for a in anno:
            a["iscrowd"] = a.get("iscrowd", 0)
            a["polygon"] = a.get("polygon", [])
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        semi = [obj.get("semi", False) for obj in anno]
        semi = torch.tensor(semi, dtype=bool)

        boxes = anno[0].get("bbox", None) if len(anno) > 0 else None
        if boxes is not None:
            boxes = [obj["bbox"] for obj in anno]
            # guard against no boxes via resizing
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]

        points = anno[0].get("point", None) if len(anno) > 0 else None
        if points is not None:
            points = [obj['point'] for obj in anno]

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # todo: FYI removed masks, segmentation and area.  Assuming these are not needed for OD
        # segmentations = [obj["segmentation"] for obj in anno if obj]
        # masks = convert_coco_poly_to_mask(segmentations, h, w)
        polygons = [obj['polygon'] for obj in anno if obj]
        polygons = [torch.tensor(p, dtype=torch.int64) for p in polygons]

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        if keep is not None:
            classes = classes[keep]
            # masks = masks[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]

        target = {}
        target["semi"] = semi
        if boxes is not None:
            target["boxes"] = boxes
        if points is not None:
            target["points"] = points
        target["labels"] = classes
        target["polygons"] = polygons
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        # target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        anno_with_bbox = [obj for obj in anno if obj.get("bbox", None) is not None]
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno_with_bbox)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(
            f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
        )
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    if len(ids) < len(dataset):
        dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets.get("iscrowd", [0] * len(targets["labels"])).tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        # Rebuild the COCO dataset to remove point annotations from dataset
        coco_dataset = dataset.coco
        return coco_dataset
    dataset = convert_to_coco_api(dataset)
    return dataset


def coco_remove_annotations(coco_dataset, keep_bbox=True, remove_semi=False, remove_all_annotations=False):
    json_dataset = coco_dataset.dataset.copy()
    if remove_all_annotations:
        json_dataset['annotations'] = []
    else:
        updated_ann = []
        for a in json_dataset["annotations"]:
            if keep_bbox and 'bbox' in a:
                # Want to add this annotation but check if it's a semi-supervised
                if not (remove_semi and a.get('semi', False)):  # Check if it's semi-supervised
                    updated_ann.append(a)
        json_dataset['annotations'] = updated_ann
    from pycocotools.coco import COCO
    coco_dataset = COCO()
    coco_dataset.dataset = json_dataset
    coco_dataset.createIndex()
    return coco_dataset


def num_annotations(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset.num_annotations()


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, is_source=True):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.is_source = is_source

    def replace_coco(self, coco):
        self.coco = coco
        self.coco.createIndex()  # Do this again just in case
        self.ids = list(sorted(self.coco.imgs.keys()))

    def num_annotations(self):
        return len(self.coco.anns)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # Used in DA training
        domain_labels = torch.ones_like(target['labels'], dtype=torch.uint8) \
            if self.is_source else torch.zeros_like(target['labels'], dtype=torch.uint8)
        target['is_source'] = domain_labels

        return img, target


def get_coco(root, image_set, transforms, is_source=True, data_pre="", mode="instances", year="2023",
             anno_file_template="{}_{}{}.json"):
    # year was 2017
    PATHS = {
        "train": (
            f"{data_pre}train{year}", os.path.join("annotations", anno_file_template.format(mode, "train", year))),
        "val": (f"{data_pre}val{year}", os.path.join("annotations", anno_file_template.format(mode, "val", year))),
        "test": (f"{data_pre}test{year}", os.path.join("annotations", anno_file_template.format(mode, "test", year)))
    }

    #t = [ConvertCocoPolysToMask()]

    #if transforms is not None:
    #    t.append(transforms)
    #transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, is_source=is_source, transforms=None)

    return dataset


def get_coco_kp(root, image_set, transforms, is_source=True):
    return get_coco(root, image_set, transforms, is_source=is_source, mode="person_keypoints")


def get_coco_cityscapes(root, image_set, transforms, is_source=True):
    image_set = "val" if image_set == "test" else image_set
    return get_coco(root, image_set, transforms, is_source=is_source,
                    mode="instancesonly_filtered_gtFine", data_pre="gtFine_")


def new_coco_dataset(category_map):
    new_dataset = COCO()
    new_dataset.dataset['info'] = {'description': 'BenthicMorphospecies',
                                   'version': 0.1,
                                   'year': 2023,
                                   'contributor': 'Heather Doig'}
    categories = []
    for i, cat in enumerate(category_map):
        categories.append({'id': i + 1, 'name': cat})
    new_dataset.dataset['info']['categories'] = categories
    new_dataset.dataset['categories'] = categories
    new_dataset.dataset['images'] = []
    new_dataset.dataset['annotations'] = []
    return new_dataset


def get_dataset(dataset_desc, transform, data_path, is_source=True):
    dataset_campaign = dataset_desc["folder"]
    dataset_type = dataset_desc["type"]
    image_set = dataset_desc["split"]
    data_path = os.path.join(data_path, dataset_campaign)
    paths = {"coco": (data_path, get_coco),
             "cityscapes": (data_path, get_coco_cityscapes)}
    p, ds_fn = paths[dataset_type]

    ds = ds_fn(p, image_set=image_set, is_source=is_source, transforms=transform)
    return ds
