from spots_detector_utils import *
from spots_detector import SpotsDetector
from sklearn.cluster import DBSCAN

JSON_PATH = "output/results.json"
WINDOW = 7
EPS = 6
MIN_SAMPLES = 5
MERGE_THRESHOLD = 5
MERGE_IOU_THRESHOLD = 0.35
INPUT_PARKING_IMAGE = "output/input/frame_5.jpg"  # only for visualization
OUTPUT_PARKING_IMAGE = "images/detected_spots_trucks.jpg"  # path to save image with spots

class SpotClusterer(SpotsDetector):
    @staticmethod
    def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        if area_a <= 0 or area_b <= 0:
            return 0.0

        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
    
    def add_frame(self, frame_index: int, detections: List[Dict[str, Any]]) -> None:
        for det in detections:
            self.records.append(
                {
                    "frame": frame_index,
                    "centroid": det["centroid"],
                    "bbox": det["bbox"],
                }
            )

    def detect_spots(self) -> List[Dict[str, Any]]:
        if not self.records:
            return []

        centroids = [record["centroid"] for record in self.records]
        clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        labels = clustering.fit_predict(centroids)

        clusters: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(index)

        candidates: List[Dict[str, Any]] = []
        for cluster_indices in clusters.values():
            frames: Set[int] = set(self.records[i]["frame"] for i in cluster_indices)
            if len(frames) < WINDOW:
                continue

            cluster_centroids = [self.records[i]["centroid"] for i in cluster_indices]
            cluster_bboxes = [self.records[i]["bbox"] for i in cluster_indices]

            xs = [c[0] for c in cluster_centroids]
            ys = [c[1] for c in cluster_centroids]

            spot = {
                "center": (sum(xs) / len(xs), sum(ys) / len(ys)),
                "bbox": self.compute_bbox(cluster_bboxes),
                "samples": len(cluster_indices),
                "frames": len(frames),
            }
            candidates.append(spot)

        merged = self.merge_close_spots(candidates)
        self.spots = merged
        return merged

    def dbscan(
        self,
        points: List[Tuple[float, float]],
        eps: float,
        min_samples: int,
    ) -> List[int]:
        n = len(points)
        labels = [-1] * n
        visited = [False] * n
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self.region_query(points, i, eps)
            if len(neighbors) < min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = [j for j in neighbors if j != i]

            while seeds:
                current = seeds.pop()
                if not visited[current]:
                    visited[current] = True
                    current_neighbors = self.region_query(points, current, eps)
                    if len(current_neighbors) >= min_samples:
                        seeds.extend([j for j in current_neighbors if j not in seeds])

                if labels[current] == -1:
                    labels[current] = cluster_id

            cluster_id += 1

        return labels
    
    @staticmethod
    def region_query(
        points: List[Tuple[float, float]],
        index: int,
        eps: float,
    ) -> List[int]:
        neighbors: List[int] = []
        for j, point in enumerate(points):
            if dist(points[index], point) <= eps:
                neighbors.append(j)
        return neighbors

    def merge_close_spots(self, spots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not spots:
            return []

        merged: List[Dict[str, Any]] = []
        used: Set[int] = set()

        for i, spot in enumerate(spots):
            if i in used:
                continue
            group = [spot]
            for j in range(i + 1, len(spots)):
                if j in used:
                    continue
                distance = dist(spot["center"], spots[j]["center"])
                overlap = self.bbox_iou(spot["bbox"], spots[j]["bbox"])
                if distance < MERGE_THRESHOLD or overlap > MERGE_IOU_THRESHOLD:
                    group.append(spots[j])
                    used.add(j)

            if len(group) == 1:
                merged.append(spot)
                continue

            centers = [s["center"] for s in group]
            bboxes = [s["bbox"] for s in group]
            samples = sum(s["samples"] for s in group)
            frames = sum(s["frames"] for s in group)

            merged.append(
                {
                    "center": (
                        sum(c[0] for c in centers) / len(centers),
                        sum(c[1] for c in centers) / len(centers),
                    ),
                    "bbox": self.compute_bbox(bboxes),
                    "samples": samples,
                    "frames": frames,
                }
            )

        return merged


def run_analysis() -> None:
    data = read_latest_json()
    if not data:
        print(f"No JSON data found at '{JSON_PATH}'.")
        return

    clusterer = SpotClusterer()

    for frame_index, frame in enumerate(data):
        detections = extract_detections(frame)
        clusterer.add_frame(frame_index, detections)

    spots = clusterer.detect_spots()

    print(f"\nCANDIDATE PARKING SPOTS num: {len(spots)}")

    image = cv2.imread(INPUT_PARKING_IMAGE)
    if image is None:
        print(f"Could not open input image '{INPUT_PARKING_IMAGE}'. Skipping visualization.")
        return

    vis = draw_parking_spots(image, spots)
    cv2.imwrite(OUTPUT_PARKING_IMAGE, vis)


if __name__ == "__main__":
    run_analysis()
