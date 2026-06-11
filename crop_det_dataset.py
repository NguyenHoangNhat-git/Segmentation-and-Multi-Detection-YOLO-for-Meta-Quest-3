import os
import cv2
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "detection_dataset_split"          
OUTPUT_ROOT = "cropped_detection_dataset_split"

# Dimensions
ORIG_W, ORIG_H = 1920, 1080
CROP_W, CROP_H = 1280, 1080  # Step 1: Crop width down, keep full height
FINAL_W, FINAL_H = 1280, 1280  # Step 2: Pad height up to square

PAD_TOP = 100  # (1280 - 1080) // 2

def process_split(split_name):
    img_dir = os.path.join(DATASET_ROOT, split_name, "images")
    label_dir = os.path.join(DATASET_ROOT, split_name, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Skipping split '{split_name}': Folders missing.")
        return

    out_img_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    out_label_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

    print(f"Processing '{split_name}' split ({len(image_files)} images)...")

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

        # Target the first bounding box center to handle the horizontal crop
        first_line = lines[0].strip().split()
        if len(first_line) != 5:
            continue
            
        _, xb_c, _, _, _ = map(float, first_line)
        obj_center_x = int(xb_c * ORIG_W)
        
        # Calculate horizontal crop boundaries (keep full 1080 height)
        crop_x1 = obj_center_x - (CROP_W // 2)
        if crop_x1 < 0: crop_x1 = 0
        if crop_x1 + CROP_W > ORIG_W: crop_x1 = ORIG_W - CROP_W
        crop_x2 = crop_x1 + CROP_W
        
        # Step 1: Crop horizontally to 1280x1080
        cropped_img = img[0:ORIG_H, crop_x1:crop_x2]
        
        # Step 2: Letterbox pad vertically to 1280x1280 (100px top, 100px bottom)
        padded_img = cv2.copyMakeBorder(
            cropped_img, 
            top=PAD_TOP, 
            bottom=PAD_TOP, 
            left=0, 
            right=0, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[0, 0, 0] # Black bars
        )
        
        new_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Convert normalized original to absolute pixels
            abs_x_c = x_c * ORIG_W
            abs_y_c = y_c * ORIG_H
            abs_w = w * ORIG_W
            abs_h = h * ORIG_H
            
            # Get bounding box limits
            abs_x1 = abs_x_c - (abs_w / 2)
            abs_y1 = abs_y_c - (abs_h / 2)
            abs_x2 = abs_x_c + (abs_w / 2)
            abs_y2 = abs_y_c + (abs_h / 2)
            
            # Move coordinates: Shift X because of crop, Shift Y because of top padding
            shifted_x1 = abs_x1 - crop_x1
            shifted_y1 = abs_y1 + PAD_TOP
            shifted_x2 = abs_x2 - crop_x1
            shifted_y2 = abs_y2 + PAD_TOP
            
            # Clip X to the cropped width boundaries
            shifted_x1 = np.clip(shifted_x1, 0, FINAL_W)
            shifted_x2 = np.clip(shifted_x2, 0, FINAL_W)
            
            # Calculate new width and height
            new_w = shifted_x2 - shifted_x1
            new_h = abs_h # Height doesn't change from cropping, padding just offsets it
            
            # Discard bounding box if it was cropped completely out of frame horizontally
            if new_w < 1:
                continue
                
            # Compute new centers in the final 1280x1280 frame
            new_x_c = shifted_x1 + (new_w / 2)
            new_y_c = shifted_y1 + (new_h / 2)
            
            # Normalize to final 1280x1280 dimensions
            norm_xc = new_x_c / FINAL_W
            norm_yc = new_y_c / FINAL_H
            norm_w = new_w / FINAL_W
            norm_h = new_h / FINAL_H
            
            new_labels.append(f"{cls} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        if new_labels:
            cv2.imwrite(os.path.join(out_img_dir, img_name), padded_img)
            with open(os.path.join(out_label_dir, label_name), 'w') as f:
                f.write("\n".join(new_labels))

    print(f"Finished processing '{split_name}' split.\n")

def main():
    for split in ["train", "val", "test"]:
        process_split(split)
    print("All splits processed successfully!")

if __name__ == "__main__":
    main()