from ultralytics import YOLO
import glob
import os

# IMAGE_COUNT = 4
IMAGE_FOLDER = "images"
ACCURACY_THRESHOLD = 0.4
TARGET_CLASSES = ["car", "truck"]
MODEL_SIZE = "l"  # n s m l x

# Ładowanie modelu (n - nano, najszybszy; x - extra large, najdokładniejszy)
model = YOLO("yolov8x.pt")


def read_images() -> list[str]:
    image_paths = glob.glob(f"{IMAGE_FOLDER}/*.png")
    image_count = len(image_paths)
    print(f"Liczba obrazów: {image_count}")
    image_paths.sort()

    return image_paths


def detect_with_bboxes(model: YOLO, image_paths: list[str], class_ids: list[int]):
    results = model.predict(
        image_paths, save=True, conf=ACCURACY_THRESHOLD, classes=class_ids
    )

    for path, result in zip(image_paths, results):
        print(f"\n=== {os.path.basename(path)} (BBOX) ===")

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]

            print(f"{name} ({conf:.2f}) -> ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


def detect_with_segmentation(model: YOLO, image_paths: list[str], class_ids: list[int]):
    results = model.predict(
        image_paths, save=True, conf=ACCURACY_THRESHOLD, classes=class_ids
    )

    for path, result in zip(image_paths, results):
        print(f"\n=== {os.path.basename(path)} (SEGMENTATION) ===")

        if result.masks is None:
            print("Brak masek")
            continue

        for i, polygon in enumerate(result.masks.xy):
            cls = int(result.boxes.cls[i])
            conf = float(result.boxes.conf[i])
            name = model.names[cls]

            print(f"{name} ({conf:.2f})")
            print(f"Wielokąt: {[(int(x), int(y)) for x, y in polygon]}")


def get_class_ids(model, class_names):
    return [k for k, v in model.names.items() if v in class_names]


def main():
    image_paths = read_images()

    class_ids = get_class_ids(model, TARGET_CLASSES)

    bbox_model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    detect_with_bboxes(bbox_model, image_paths, class_ids)

    seg_model = YOLO(f"yolov8{MODEL_SIZE}-seg.pt")
    detect_with_segmentation(seg_model, image_paths, class_ids)


if __name__ == "__main__":
    main()
