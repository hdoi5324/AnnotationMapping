from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from .schema import AnnRecord, ImageRecord


def _new_coco_dataset(category_name_list: Sequence[str]) -> dict[str, Any]:
    dataset: dict[str, Any] = {}
    dataset["info"] = {"description": "BenthicMorphospecies", "contributor": "Heather Doig"}
    categories = [{"id": i + 1, "name": cat} for i, cat in enumerate(category_name_list)]
    dataset["info"]["categories"] = categories
    dataset["categories"] = categories
    dataset["images"] = []
    dataset["annotations"] = []
    return dataset


def _as_image_record(img: ImageRecord | Mapping[str, Any]) -> ImageRecord:
    if isinstance(img, ImageRecord):
        return img
    return ImageRecord(
        file_name=str(img["file_name"]),
        width=int(img["width"]),
        height=int(img["height"]),
        source_id=img.get("id"),
        extra={k: v for k, v in img.items() if k not in {"file_name", "width", "height", "id"}},
    )


def _as_ann_record(ann: AnnRecord | Mapping[str, Any]) -> AnnRecord:
    if isinstance(ann, AnnRecord):
        return ann
    polygon = ann.get("polygon")
    if polygon is not None and len(polygon) < 3:
        polygon = None
    return AnnRecord(
        category_id=int(ann.get("category") if "category" in ann else ann["category_id"]),
        bbox=ann.get("bbox"),
        point=ann.get("point"),
        polygon=polygon,
        semi=bool(ann.get("semi", False)),
        extra={k: v for k, v in ann.items() if k not in {"category", "category_id", "bbox", "point", "polygon", "semi"}},
    )


class CocoWriter:
    """
    Minimal COCO JSON writer.

    Backwards compatible with the existing dict-based records in this repo.
    """

    def __init__(self, categories: Sequence[str], id_policy: str = "sequential"):
        self._dataset = _new_coco_dataset(list(categories))
        self._id_policy = id_policy
        self._next_image_id = 1
        self._next_ann_id = 1

    @property
    def dataset(self) -> dict[str, Any]:
        return self._dataset

    def load_existing(self, path: str) -> None:
        with open(path, "r") as f:
            self._dataset = json.load(f)
        images = self._dataset.get("images", []) or []
        anns = self._dataset.get("annotations", []) or []
        self._next_image_id = (max((i.get("id", 0) for i in images), default=0) + 1) or 1
        self._next_ann_id = (max((a.get("id", 0) for a in anns), default=0) + 1) or 1

    def add_image(self, image: ImageRecord | Mapping[str, Any]) -> int:
        img = _as_image_record(image)
        if self._id_policy == "source" and isinstance(img.source_id, int):
            image_id = int(img.source_id)
        else:
            image_id = self._next_image_id
            self._next_image_id += 1

        img_dict: dict[str, Any] = {
            "file_name": img.file_name,
            "height": int(img.height),
            "width": int(img.width),
            "id": image_id,
        }
        if img.source_id is not None:
            img_dict["source_id"] = img.source_id
        if img.extra:
            img_dict.update(dict(img.extra))

        self._dataset.setdefault("images", []).append(img_dict)
        return image_id

    def add_annotations(self, image_id: int, annotations: Iterable[AnnRecord | Mapping[str, Any]]) -> None:
        for a in annotations:
            ann = _as_ann_record(a)
            ann_dict: dict[str, Any] = {
                "category_id": int(ann.category_id),
                "image_id": int(image_id),
                "id": int(self._next_ann_id),
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": [],
                "semi": bool(ann.semi),
            }
            self._next_ann_id += 1

            if ann.bbox is not None:
                bbox_xywh = [int(b) for b in ann.bbox]
                ann_dict["bbox"] = bbox_xywh
                ann_dict["area"] = int(bbox_xywh[2] * bbox_xywh[3])
            if ann.point is not None:
                ann_dict["point"] = [int(ann.point[0]), int(ann.point[1])]
            if ann.polygon is not None and len(ann.polygon) >= 3:
                ann_dict["polygon"] = [list(map(int, p)) for p in ann.polygon]

            if ann.extra:
                ann_dict.update(dict(ann.extra))

            self._dataset.setdefault("annotations", []).append(ann_dict)

    def dump(self, path: str) -> None:
        with open(path, "w") as fp:
            json.dump(self._dataset, fp)

