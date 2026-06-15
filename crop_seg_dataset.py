
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
# Path to your main dataset directory containing train, val, and test folders
DATASET_ROOT = "segmentation_dataset_split"  
OUTPUT_ROOT = "cropped_segmentation_dataset_split"

CROP_W, CROP_H = 640, 640
ORIG_W, ORIG_H = 1920, 1080

def process_split(split_name):
    """Processes a specific split (e.g., 'train', 'val', 'test')"""
    img_dir = os.path.join(DATASET_ROOT, split_name, "images")
    label_dir = os.path.join(DATASET_ROOT, split_name, "labels")
    
    # Check if this split actually exists in the source directory
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Skipping split '{split_name}': 'images' or 'labels' folder not found.")
        return

    # Define and create output directories for this split
    out_img_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    out_label_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

    print(f"Processing '{split_name}' split ({len(image_files)} images found)...")

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue

        # Target the first object in the file to center the crop window on
        first_line = lines[0].strip().split()
        if len(first_line) < 7: 
            continue
            
        class_id = first_line[0]
        coords = np.array([float(x) for x in first_line[1:]]).reshape(-1, 2)
        
        # Absolute pixel conversion
        pixel_coords = coords * [ORIG_W, ORIG_H]
        
        # Center point bounding box calculation
        x_min, y_min = np.min(pixel_coords, axis=0)
        x_max, y_max = np.max(pixel_coords, axis=0)
        obj_center_x = int((x_min + x_max) / 2)
        obj_center_y = int((y_min + y_max) / 2)
        
        # Top-left corner box placement
        crop_x1 = obj_center_x - (CROP_W // 2)
        crop_y1 = obj_center_y - (CROP_H // 2)
        
        # Frame boundary guardrails
        if crop_x1 < 0: crop_x1 = 0
        if crop_y1 < 0: crop_y1 = 0
        if crop_x1 + CROP_W > ORIG_W: crop_x1 = ORIG_W - CROP_W
        if crop_y1 + CROP_H > ORIG_H: crop_y1 = ORIG_H - CROP_H
            
        crop_x2 = crop_x1 + CROP_W
        crop_y2 = crop_y1 + CROP_H
        
        # Perform image crop
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Recalculate coordinates for all instances present in the crop frame
        new_labels = []
        for line in lines:
            parts = line.strip().split()
            cls = parts[0]
            poly_coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            
            abs_poly = poly_coords * [ORIG_W, ORIG_H]
            shifted_poly = abs_poly - [crop_x1, crop_y1]
            
            # Clip overflowing mask edges to boundary lines
            shifted_poly[:, 0] = np.clip(shifted_poly[:, 0], 0, CROP_W)
            shifted_poly[:, 1] = np.clip(shifted_poly[:, 1], 0, CROP_H)
            
            poly_x_min, poly_y_min = np.min(shifted_poly, axis=0)
            poly_x_max, poly_y_max = np.max(shifted_poly, axis=0)
            
            if (poly_x_max - poly_x_min) < 1 or (poly_y_max - poly_y_min) < 1:
                continue
                
            norm_poly = shifted_poly / [CROP_W, CROP_H]
            flat_coords = " ".join([f"{coord:.6f}" for pair in norm_poly for coord in pair])
            new_labels.append(f"{cls} {flat_coords}")
            
        # Write to target files if crop successfully contains labels
        if new_labels:
            out_img_path = os.path.join(out_img_dir, img_name)
            out_label_path = os.path.join(out_label_dir, label_name)
            
            cv2.imwrite(out_img_path, cropped_img)
            with open(out_label_path, 'w') as f:
                f.write("\n".join(new_labels))

    print(f"Finished processing '{split_name}' split.\n")

def main():
    # Standard YOLO format folders
    dataset_splits = ["train", "val", "test"]
    
    for split in dataset_splits:
        process_split(split)
        
    print("Full dataset conversion complete!")

if __name__ == "__main__":
    main()