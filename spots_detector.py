import json
import time
from collections import defaultdict, deque
import math
import cv2
from typing import List, Dict, Tuple, Any, Optional

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


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def read_latest_json() -> List[Dict[str, Any]]:
    try:
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    except:
        return []


def extract_detections(frame_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    for det in frame_json["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        cx: float = (x1 + x2) / 2
        cy: float = (y1 + y2) / 2

        detections.append({"centroid": (cx, cy), "bbox": (x1, y1, x2, y2)})

    return detections


def draw_parking_spots(
    image: Any, spots: List[Dict[str, Any]], color: Tuple[int, int, int] = (0, 255, 0)
) -> Any:

    output = image.copy()

    for i, spot in enumerate(spots):

        x1, y1, x2, y2 = map(int, spot["bbox"])

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cx, cy = map(int, spot["center"])
        cv2.circle(output, (cx, cy), 4, color, -1)

        cv2.putText(
            output,
            "",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def run_analysis() -> None:
    tracker: PointTracker = PointTracker()
    last_len: int = 0

    try:
        with open(JSON_PATH, "r") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except:
        time.sleep(1)
        return

    new_data: List[Dict[str, Any]] = data[last_len:]
    last_len = len(data)

    for frame in new_data:
        detections = extract_detections(frame)
        tracker.update(detections)

    spots = tracker.detect_spots()

    print(f"\nCANDIDATE PARKING SPOTS num: {len(spots)}")

    vis = draw_parking_spots(cv2.imread(INPUT_PARKING_IMAGE), spots)

    cv2.imwrite(OUTPUT_PARKING_IMAGE, vis)


class PointTracker:
    def __init__(self) -> None:
        self.tracks: List[Dict[str, Any]] = []
        self.spots: List[Dict[str, Any]] = []

    @staticmethod
    def compute_bbox(
        bboxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:

        xs1 = [b[0] for b in bboxes]
        ys1 = [b[1] for b in bboxes]
        xs2 = [b[2] for b in bboxes]
        ys2 = [b[3] for b in bboxes]

        return (min(xs1), min(ys1), max(xs2), max(ys2))

    def update(self, detections: List[Dict[str, Any]]) -> None:

        for d in detections:
            centroid: Tuple[float, float] = d["centroid"]
            bbox: Tuple[float, float, float, float] = d["bbox"]

            matched: bool = False

            for t in self.tracks:
                if dist(centroid, t["last_center"]) < DIST_THRESHOLD:
                    t["centroids"].append(centroid)
                    t["bboxes"].append(bbox)
                    t["last_center"] = centroid
                    matched = True
                    break

            if not matched:
                self.tracks.append(
                    {"centroids": [centroid], "bboxes": [bbox], "last_center": centroid}
                )

    def detect_spots(self) -> List[Dict[str, Any]]:

        candidates: List[Dict[str, Any]] = []

        for t in self.tracks:

            if len(t["centroids"]) < WINDOW:
                continue

            window_centroids = t["centroids"][-WINDOW:]
            window_bboxes = t["bboxes"][-WINDOW:]

            stable: bool = True

            for i in range(1, len(window_centroids)):
                if dist(window_centroids[i], window_centroids[i - 1]) > DIST_THRESHOLD:
                    stable = False
                    break

            if stable:

                xs = [p[0] for p in window_centroids]
                ys = [p[1] for p in window_centroids]

                bbox = self.compute_bbox(window_bboxes)

                spot: Dict[str, Any] = {
                    "center": (sum(xs) / len(xs), sum(ys) / len(ys)),
                    "bbox": bbox,
                    "samples": len(window_centroids),
                }

                candidates.append(spot)

        self.spots = candidates
        return candidates


if __name__ == "__main__":
    run_analysis()
