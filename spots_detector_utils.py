from typing import List, Dict, Tuple, Any, Set
import json
import cv2
import math

JSON_PATH = "output/results.json"

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


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
