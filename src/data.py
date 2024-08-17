from dataclasses import dataclass
from typing import List

@dataclass
class BoundingBox2D:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_label: str

@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float
    confidence: float
    class_label: str

@dataclass
class Mask:
    mask_data: List[int]
    width: int
    height: int

@dataclass
class KeyPoint2D:
    x: float
    y: float
    confidence: float

@dataclass
class KeyPoint3D:
    x: float
    y: float
    z: float
    confidence: float

@dataclass
class YoloDetectionResult:
    class_id: int
    class_name: str
    score: float
    id: str
    bbox: BoundingBox2D
    bbox3d: BoundingBox3D
    mask: Mask
    keypoints: List[KeyPoint2D]
    keypoints3d: List[KeyPoint3D]