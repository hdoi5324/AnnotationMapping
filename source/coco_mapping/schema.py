from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    width: int
    height: int
    source_id: int | str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnRecord:
    category_id: int
    bbox: Sequence[int] | None = None  # COCO xywh
    point: Sequence[int] | None = None  # [x, y] in pixels
    polygon: Sequence[Sequence[int]] | None = None  # [[x,y], ...] in pixels
    semi: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

