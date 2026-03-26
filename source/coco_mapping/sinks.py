from __future__ import annotations

from typing import Iterable, Mapping

from .schema import AnnRecord


def _as_ann_dict(a: AnnRecord | Mapping) -> Mapping:
    return a if isinstance(a, Mapping) else {
        "category": a.category_id,
        "bbox": a.bbox,
        "point": a.point,
        "polygon": a.polygon,
        "semi": a.semi,
    }


def write_box_txt(img_path: str, annotations: Iterable[AnnRecord | Mapping]) -> None:
    text_lines: list[str] = []
    for a in annotations:
        d = _as_ann_dict(a)
        if "bbox" in d and d["bbox"] is not None:
            bbox = d["bbox"]
            cat = d.get("category") if "category" in d else d.get("category_id")
            text_lines.append(f"{int(cat)}, {int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}\n")

    if not text_lines:
        return
    box_path = f"{img_path[:-4]}.txt"
    with open(box_path, "w+") as f:
        for line in text_lines:
            f.write(line)


def write_viz(img_path: str, annotations: Iterable[AnnRecord | Mapping]) -> None:
    # Imported lazily so core generation doesn't require these deps.
    import cv2
    import matplotlib as mpl
    import numpy as np

    anns = list(annotations)
    if not anns:
        return

    output_name = f"{img_path[:-4]}_viz.jpg"
    image = cv2.imread(img_path)
    if image is None:
        return

    colours = [
        (int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))
        for c in mpl.colormaps["tab10"].colors
    ]
    for a in anns:
        d = _as_ann_dict(a)
        cat = int(d.get("category") if "category" in d else d.get("category_id", 0))
        semi = bool(d.get("semi", False))
        colour = colours[cat % len(colours)] if not semi else colours[-1]

        bbox = d.get("bbox")
        point = d.get("point")
        if bbox is not None:
            x, y, w, h = bbox
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + int(w), y0 + int(h)
            image = cv2.rectangle(image, (x0, y0), (x1, y1), colour, 4)
        elif point is not None:
            x, y = int(point[0]), int(point[1])
            image = cv2.circle(image, (x, y), 8, colour, 16)

        polygon = d.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            image = cv2.drawContours(image, [np.array(polygon)], 0, (255, 255, 255), 1)

    cv2.imwrite(output_name, image)

