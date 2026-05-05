from abc import abstractmethod
import json
import time
from collections import defaultdict, deque
import math
import cv2
from typing import List, Dict, Tuple, Any, Optional
from spots_detector_utils import *

JSON_PATH = "output/results.json"
WINDOW = 10
DIST_THRESHOLD = 15
INPUT_PARKING_IMAGE = "images/frame_7.jpg"  # only for vizualization
OUTPUT_PARKING_IMAGE = (
    "images/detected_spots_trucks.jpg"  # path to save image with spots
)

# TODO implementation for polygons
# TODO maybe some better algorithm for clustering instead of simple dist_threshold
# TODO some automatic postprocessing of parking spots -> now situation can happen when car is present on the spot in one time window and then
#       after a while some other car is present for a while and it is calculated as two separate spots.
#       Probably this could be taken care of by clustering at the end

class SpotsDetector:
    def __init__(self):
        self.spots = []
        self.tracks = []
        self.records = []
    
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def detect_spots(self) -> List[Dict[str, Any]]:
        pass

    @staticmethod
    def compute_bbox(
        bboxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:
        xs1 = [b[0] for b in bboxes]
        ys1 = [b[1] for b in bboxes]
        xs2 = [b[2] for b in bboxes]
        ys2 = [b[3] for b in bboxes]

        return (min(xs1), min(ys1), max(xs2), max(ys2))
