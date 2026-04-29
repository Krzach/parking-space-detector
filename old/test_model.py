import os
import random
from ultralytics import YOLO

def main():
    # Path to the best weights generated after training
    weights_path = 'runs/detect/parking_space_detector/yolov8_training/weights/best.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please ensure you have completed the training first.")
        return

    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    # Let's test on a random validation image we prepared earlier
    val_images_dir = 'yolo_dataset/images/val'
    if not os.path.exists(val_images_dir):
        print(f"Error: Validation images not found at {val_images_dir}")
        return
        
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')]
    if not val_images:
        print("No validation images found to test on.")
        return
        
    # Pick a random image
    test_image_name = random.choice(val_images)
    test_image_path = os.path.join(val_images_dir, test_image_name)
    
    print(f"\nRunning inference on: {test_image_path}")
    
    # Run inference
    # save=True will save the annotated image to the runs/detect/predict folder
    results = model.predict(source=test_image_path, save=True, conf=0.25)
    
    print("\nInference complete!")
    print(f"The resulting image with bounding boxes has been saved to: {results[0].save_dir}")

if __name__ == '__main__':
    main()
