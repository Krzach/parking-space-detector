import os
import xml.etree.ElementTree as ET
import random
from PIL import Image
from ultralytics import YOLO

def convert_pklot_to_yolo(dataset_base, yolo_dataset_path, max_images=5000):
    """
    Parses the PKLot XML annotations and converts them into YOLO format.
    Creates symlinks to the images to save disk space.
    """
    if os.path.exists(yolo_dataset_path):
        print(f"YOLO dataset already found at {yolo_dataset_path}. Skipping conversion.")
        return yolo_dataset_path
        
    os.makedirs(os.path.join(yolo_dataset_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dataset_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dataset_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dataset_path, 'labels', 'val'), exist_ok=True)
    
    print("Scanning dataset for XML annotations...")
    xml_files = []
    for dir_root, _, file_names in os.walk(dataset_base):
        for file in file_names:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(dir_root, file))
                
    print(f"Found {len(xml_files)} XML files.")
    
    # We sample the dataset because training on 600,000+ images would take days/weeks.
    # You can increase 'max_images' or set it to len(xml_files) if you want to use the entire dataset.
    random.seed(42)
    sample_size = min(len(xml_files), max_images)
    if sample_size < len(xml_files):
        print(f"Sampling {sample_size} images for practical training time...")
    xml_files = random.sample(xml_files, sample_size)
    
    # 80/20 train/val split
    split_idx = int(len(xml_files) * 0.8)
    train_xmls = xml_files[:split_idx]
    val_xmls = xml_files[split_idx:]
    
    def process_split(files_list, split_name):
        processed_count = 0
        for i, xml_file in enumerate(files_list):
            jpg_file = xml_file.replace('.xml', '.jpg')
            if not os.path.exists(jpg_file):
                continue
                
            try:
                tree = ET.parse(xml_file)
                xml_root = tree.getroot()
                with Image.open(jpg_file) as img:
                    img_width, img_height = img.size
            except Exception:
                continue
                
            yolo_lines = []
            for space in xml_root.findall('space'):
                occupied = space.get('occupied')
                # class 0 = free, class 1 = taken
                class_id = 1 if occupied == '1' else 0
                
                contour = space.find('contour')
                if contour is not None:
                    xs = [int(pt.get('x')) for pt in contour.findall('point')]
                    ys = [int(pt.get('y')) for pt in contour.findall('point')]
                    if not xs or not ys: 
                        continue
                    
                    # Compute axis-aligned bounding box from contour
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    # Normalize for YOLO format (center_x, center_y, width, height)
                    w = (max_x - min_x) / img_width
                    h = (max_y - min_y) / img_height
                    cx = (min_x + max_x) / 2.0 / img_width
                    cy = (min_y + max_y) / 2.0 / img_height
                    
                    # Clamp values between 0.0 and 1.0 just in case
                    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
                    w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
                    
                    yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            if not yolo_lines:
                continue
                
            # Create unique filenames to avoid collision across different directories
            base_name = f"{split_name}_{i}_{os.path.basename(jpg_file)}"
            dest_jpg = os.path.join(yolo_dataset_path, 'images', split_name, base_name)
            dest_txt = os.path.join(yolo_dataset_path, 'labels', split_name, base_name.replace('.jpg', '.txt'))
            
            # Use symlinks to avoid copying hundreds of megabytes/gigabytes of image data
            os.symlink(os.path.abspath(jpg_file), dest_jpg)
            
            with open(dest_txt, 'w') as out_f:
                out_f.write('\n'.join(yolo_lines))
                
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"[{split_name}] Processed {processed_count} files...")

    print("--- Processing training files ---")
    process_split(train_xmls, 'train')
    print("--- Processing validation files ---")
    process_split(val_xmls, 'val')
    
    # Generate data.yaml for YOLO
    yaml_content = f"""path: {os.path.abspath(yolo_dataset_path)}
train: images/train
val: images/val

names:
  0: free
  1: taken
"""
    with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'w') as yaml_f:
        yaml_f.write(yaml_content)
        
    return yolo_dataset_path

def main():
    # The dataset contains two main folders: PKLot (full images) and PKLotSegmented (individual spots)
    # Since we want to detect on a full picture, we should use the full images in 'PKLot' folder.
    dataset_base = os.path.abspath('pklot/PKLot/PKLot')
    yolo_dataset_path = os.path.abspath('yolo_dataset')
    
    if not os.path.exists(dataset_base):
        print(f"Error: Could not find the dataset at {dataset_base}")
        return

    print("Step 1: Preparing YOLO dataset from PKLot XML annotations...")
    # I set max_images to 5000 so the script runs in a reasonable time.
    # The actual dataset has >10,000 images, you can remove max_images or set it higher to use all.
    convert_pklot_to_yolo(dataset_base, yolo_dataset_path, max_images=5000)
    
    data_yaml = os.path.join(yolo_dataset_path, 'data.yaml')
    
    print("\nStep 2: Loading YOLOv8 model...")
    # yolov8n.pt is the Nano model. It is very fast and a great starting point.
    model = YOLO('yolov8n.pt')
    
    print("\nStep 3: Starting training...")
    model.train(
        data=data_yaml,
        epochs=15,               # Increase epochs for better accuracy (e.g., 50-100)
        imgsz=640,               # Image size to resize to before passing to network
        batch=16,                # Adjust based on your GPU VRAM
        project='parking_space_detector',
        name='yolov8_training'
    )
    
    print("\nTraining complete! Your model weights are saved in 'parking_space_detector/yolov8_training/weights'.")
    print("You can test it on an image using:")
    print("yolo predict model=parking_space_detector/yolov8_training/weights/best.pt source=your_image.jpg")

if __name__ == '__main__':
    main()
