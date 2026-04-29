from ultralytics import YOLO
import cv2
import time
import os
import json
from typing import List, Dict, Any, Optional

# =========================
# CONFIG
# =========================
STREAM_URL = "https://s53.nysdot.skyvdn.com/rtplive/TA_258/playlist.m3u8"
ACCURACY_THRESHOLD = 0.2
TARGET_CLASSES = ["car", "truck"]
MODEL_SIZE = "m"  # n s m l x
INTERVAL = 30
OUTPUT_DIR = "output"
USE_SEGMENTATION = True

INPUT_DIR = f"{OUTPUT_DIR}/input"
OUTPUT_IMG_DIR = f"{OUTPUT_DIR}/predicted"
JSON_PATH = f"{OUTPUT_DIR}/results.json"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def get_class_ids(model: YOLO, class_names: List[str]) -> List[int]:
    return [k for k, v in model.names.items() if v in class_names]


def save_json(data: Dict[str, Any]) -> None:
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            existing: List[Dict[str, Any]] = json.load(f)
    else:
        existing = []

    existing.append(data)

    with open(JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def process_frame(
    model: YOLO, frame: Any, frame_id: int, class_ids: List[int], use_seg: bool = False
) -> None:

    results = model.predict(frame, conf=ACCURACY_THRESHOLD, classes=class_ids)

    result = results[0]

    input_path: str = f"{INPUT_DIR}/frame_{frame_id}.jpg"
    cv2.imwrite(input_path, frame)

    annotated = result.plot()
    output_path: str = f"{OUTPUT_IMG_DIR}/frame_{frame_id}.jpg"
    cv2.imwrite(output_path, annotated)

    detections: List[Dict[str, Any]] = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf: float = float(box.conf[0])
            cls: int = int(box.cls[0])
            name: str = model.names[cls]

            item: Dict[str, Any] = {
                "class": name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }

            if use_seg and result.masks is not None:
                if i < len(result.masks.xy):
                    polygon = result.masks.xy[i]
                    item["polygon"] = [(float(x), float(y)) for x, y in polygon]

            detections.append(item)

    save_json(
        {
            "frame_id": frame_id,
            "input_image": input_path,
            "output_image": output_path,
            "detections": detections,
        }
    )

    print(f"Frame {frame_id}: {len(detections)} objects")


def main() -> None:

    model_path: str = f"yolov8{MODEL_SIZE}"
    if USE_SEGMENTATION:
        model_path += "-seg"

    model: YOLO = YOLO(model_path + ".pt")

    class_ids: List[int] = get_class_ids(model, TARGET_CLASSES)

    cap: cv2.VideoCapture = cv2.VideoCapture(STREAM_URL)

    last_time: float = 0
    frame_id: int = 0

    while True:
        ret: bool
        frame: Any
        ret, frame = cap.read()

        if not ret:
            print("Stream disconnected, retrying...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        now: float = time.time()

        if now - last_time >= INTERVAL:
            last_time = now

            print(f"\n=== FRAME {frame_id} ===")

            process_frame(
                model=model,
                frame=frame,
                frame_id=frame_id,
                class_ids=class_ids,
                use_seg=USE_SEGMENTATION,
            )

            frame_id += 1


if __name__ == "__main__":
    main()
