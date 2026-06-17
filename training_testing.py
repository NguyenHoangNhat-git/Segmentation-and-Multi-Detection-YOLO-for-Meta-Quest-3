from ultralytics import YOLO
import os

def train_model(model_variant, proj_root, epochs=100, imgsz=640, mode="detect", run_name="yolov26n_run"):
    """
    Trains a YOLO model (Detection or Segmentation).
    data_yaml: Path to your dataset config (e.g., 'data.yaml')
    """
    dataset_root = os.path.abspath(proj_root)
    yaml_path = os.path.join(dataset_root, 'data.yaml')

    yaml_content = f"""
    path: {dataset_root}
    train: train/images
    val: val/images
    test: test/images

    nc: 16
    names: ['Bowl', 'CanOfCocaCola', 'FryingPan', 'Glass', 'Jam', 'Lid', 'MilkBottle', 'Mug', 'OilBottle', 'Plate', 'Rice', 'Saucepan', 'Sponge', 'Sugar', 'VinegarBottle', 'WashLiquid']

    """

    with open(yaml_path, 'w') as f:

        f.write(yaml_content) 
    model = YOLO(model_variant)
    
    # Train
    print(f"--- Starting Training: {model_variant} ---")
    if mode == "detect":
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            device=[0],
            
            # --- Augmentation Hyperparameters ---
            hsv_h=0.015,   # Image HSV-Hue augmentation (fraction)
            hsv_s=0.7,     # Image HSV-Saturation augmentation (fraction)
            hsv_v=0.4,     # Image HSV-Value augmentation (fraction)
            degrees=10.0,  # Image rotation (+/- deg)
            translate=0.1, # Image translation (+/- fraction)
            scale=0.5,     # Image scale (+/- gain)
            shear=2.0,     # Image shear (+/- deg)
            perspective=0.0, # Image perspective (+/- fraction), range 0-0.001
            flipud=0.0,    # Image flip up-down (probability)
            fliplr=0.5,    # Image flip left-right (probability)
            mosaic=1.0,    # Image mosaic (probability)
            mixup=0.1,     # Image mixup (probability)
            copy_paste=0.1, # Segment copy-paste (probability)
            name=run_name,
        )
    elif mode == "segment":
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,            # Start here, increase to 32 if VRAM allows
            device=[0],
            
            # --- Segmentation Specifics ---
            mask_ratio=1,        # Better mask resolution for precise grasping
            overlap_mask=True,   # Handle overlapping/stacked objects
            
            # --- Enhanced Augmentations ---
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,        # Slightly more rotation help for top-down grasping
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.3       # Increased for segmentation benefit
        )
    elif mode == "semantic":
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,          
            device=[0],
            name=run_name,
            resume=False,
            task="semantic", 
            
            # --- Augmentations ---
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.3
        )
    return results

def test_model(weights_path, source_path):
    """
    Runs inference (testing) using trained weights.
    """
    model = YOLO(weights_path)
    print(f"--- Running Validation/Test ---")
    metrics = model.val() 
    results = model.predict(source=source_path, save=True, conf=0.25)
    
    return metrics, results

# YOLOv8 Nano
train_model('yolov8n.pt', 'cropped_detection_dataset_split', run_name="yolov8n_crop_run", imgsz=1280, mode="detect")
test_model('runs/detect/train/weights/best.pt', 'cropped_detection_dataset_split/test')

# YOLO26 Nano Semantic Segmentation
train_model('yolov26n-sem.pt', 'cropped_detection_dataset_split', run_name="yolov26n_semantic_crop_run", imgsz=1280, mode="semantic")
test_model('runs/semantic/yolo26_run/weights/best.pt', 'cropped_segmentation_dataset_split/test')