import os
import shutil

def convert_and_migrate(src_root, dst_root):
    # Define subdirectories
    src_images = os.path.join(src_root, 'images')
    src_labels = os.path.join(src_root, 'labels')
    dst_images = os.path.join(dst_root, 'images')
    dst_labels = os.path.join(dst_root, 'labels')

    # Create destination directories
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    label_files = [f for f in os.listdir(src_labels) if f.endswith('.txt')]

    for filename in label_files:
        # 1. Process and Convert Labels
        with open(os.path.join(src_labels, filename), 'r') as f:
            lines = f.readlines()

        new_bboxes = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: continue
            
            class_id = int(parts[0])
            coords = parts[1:]
            
            # Extract x and y, then find min/max
            xs, ys = coords[0::2], coords[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Calculate YOLO detection format
            w, h = max_x - min_x, max_y - min_y
            cx, cy = min_x + (w / 2), min_y + (h / 2)
            
            new_bboxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save converted label
        with open(os.path.join(dst_labels, filename), 'w') as f:
            f.write('\n'.join(new_bboxes))

        # 2. Copy corresponding image
        # Checking for common extensions as YOLO usually pairs .txt with .jpg/.png
        base_name = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.png']:
            img_name = base_name + ext
            src_img_path = os.path.join(src_images, img_name)
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, os.path.join(dst_images, img_name))
                break

    print(f"Success! Detection dataset created at: {dst_root}")

# Run the script
source = 'grasping-in-the-wild.segmentation/train'
destination = 'grasping-in-the-wild.detection/train'

convert_and_migrate(source, destination)