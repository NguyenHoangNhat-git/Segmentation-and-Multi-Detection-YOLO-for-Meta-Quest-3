import os
import cv2
import numpy as np
import random

# --- CONFIGURATION ---
# Point this to the newly generated 640x640 dataset
DATASET_ROOT = "cropped_segmentation_dataset_split"
SPLIT = "train"  # Change to 'val' or 'test' if desired
NUM_SAMPLES = 5  # Number of random images you want to inspect

CROP_W, CROP_H = 640, 640

def visualize_dataset():
    img_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    label_dir = os.path.join(DATASET_ROOT, SPLIT, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Error: Could not find directories in {os.path.join(DATASET_ROOT, SPLIT)}")
        return

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    
    if not image_files:
        print(f"No images found in {img_dir}")
        return

    # Select random samples (or fewer if the dataset is small)
    samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    
    print(f"Displaying {len(samples)} random images from '{SPLIT}' split.")
    print("-> Press ANY KEY to go to the next image.")
    print("-> Press 'q' to quit.")

    for img_name in samples:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Create an overlay layer for translucent masks
        overlay = img.copy()
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                
                # Extract coordinates and reshape
                poly_coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                
                # Denormalize coordinates to 640x640 boundaries
                pixel_coords = (poly_coords * [CROP_W, CROP_H]).astype(np.int32)
                
                # Generate a random distinct color for each separate mask instance
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                
                # Draw filled polygon on the overlay
                cv2.fillPoly(overlay, [pixel_coords], color)
                # Draw a solid outline around the polygon
                cv2.polylines(img, [pixel_coords], isClosed=True, color=color, thickness=2)
        
        # Blend the original image and the mask overlay together (alpha = opacity value)
        alpha = 0.4
        visualized_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Display the window
        cv2.imshow("YOLO Segmentation Check (640x640)", visualized_img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Visualization closed.")

if __name__ == "__main__":
    visualize_dataset()