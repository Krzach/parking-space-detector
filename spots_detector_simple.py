from spots_detector_utils import *
from spots_detector import SpotsDetector
import time


JSON_PATH = "output/results.json"
WINDOW = 10
DIST_THRESHOLD = 15
INPUT_PARKING_IMAGE = "output/input/frame_5.jpg"  # only for vizualization
OUTPUT_PARKING_IMAGE = (
    "images/detected_spots_trucks_simple.jpg"  # path to save image with spots
)

class PointTracker(SpotsDetector):
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

if __name__ == "__main__":
    run_analysis()
