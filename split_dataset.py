import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(src_root, dst_root, train_size=0.7, val_size=0.15, test_size=0.15):
    # Ensure sizes add up to 1
    assert train_size + val_size + test_size == 1.0, "Split sizes must sum to 1"

    # Setup paths
    images_src = os.path.join(src_root, 'images')
    labels_src = os.path.join(src_root, 'labels')
    
    # Get all base filenames (without extension) from labels
    filenames = [os.path.splitext(f)[0] for f in os.listdir(labels_src) if f.endswith('.txt')]
    
    # Split into Train and "The Rest" (Val + Test)
    train_files, rest_files = train_test_split(filenames, train_size=train_size, random_state=42)
    
    # Split "The Rest" into Val and Test
    # Calculate relative size: 0.1 is half of the remaining 0.2
    relative_val_size = val_size / (val_size + test_size)
    val_files, test_files = train_test_split(rest_files, train_size=relative_val_size, random_state=42)

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        # Create directories
        img_dst = os.path.join(dst_root, split_name, 'images')
        lab_dst = os.path.join(dst_root, split_name, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lab_dst, exist_ok=True)

        for name in files:
            # Move Label
            shutil.copy2(os.path.join(labels_src, f"{name}.txt"), os.path.join(lab_dst, f"{name}.txt"))
            
            # Move Image (checking common extensions)
            found_img = False
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(images_src, f"{name}{ext}")
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(img_dst, f"{name}{ext}"))
                    found_img = True
                    break
            
            if not found_img:
                print(f"Warning: Image for {name} not found.")

    print(f"Dataset split complete!")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# Configuration
# source_dir_seg = 'grasping-in-the-wild.segmentation/train' 
# final_dir_seg = 'segmentation_dataset_split'
# split_dataset(source_dir_seg, final_dir_seg)

source_dir_detect = 'grasping-in-the-wild.detection/train' 
final_dir_detect = 'segmentation_dataset_split'

split_dataset(source_dir_detect, final_dir_detect)