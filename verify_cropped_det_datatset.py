import os
import cv2
import random

# --- CONFIGURATION ---
DATASET_ROOT = "cropped_detection_dataset_split"
SPLIT = "train"
NUM_SAMPLES = 5

FINAL_W, FINAL_H = 1280, 1280

def verify_padded_dataset():
    img_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    label_dir = os.path.join(DATASET_ROOT, SPLIT, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print("Error: Target output directories do not exist.")
        return

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    
    if not image_files:
        print("No converted images found to verify.")
        return

    samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    print(f"Showing {len(samples)} samples. Press 'q' to exit, or any other key for next image.")

    for img_name in samples:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls, x_c, y_c, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Denormalize bounding box coordinates back to 1280x1280 boundaries
                abs_xc = x_c * FINAL_W
                abs_yc = y_c * FINAL_H
                abs_w = w * FINAL_W
                abs_h = h * FINAL_H
                
                x1 = int(abs_xc - (abs_w / 2))
                y1 = int(abs_yc - (abs_h / 2))
                x2 = int(abs_xc + (abs_w / 2))
                y2 = int(abs_yc + (abs_h / 2))
                
                # Draw a bright green box around the object
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"Class: {cls}", (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Downscale visual window to fit comfortably on most standard desktop monitors
        cv2.namedWindow("YOLO Letterbox Check (1280x1280)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Letterbox Check (1280x1280)", 800, 800)
        cv2.imshow("YOLO Letterbox Check (1280x1280)", img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_padded_dataset()