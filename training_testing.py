from ultralytics import YOLO
import os

def train_model(model_variant, proj_root, epochs=100, imgsz=640, mode="detect", run_name="yolov9t_run"):
    """
    Trains a YOLO model (Detection or Segmentation).
    model_variant: 'yolov8n.pt' or 'yolo26n-seg.pt'
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

    # Load the model
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
    return results

def test_model(weights_path, source_path):
    """
    Runs inference (testing) using trained weights.
    weights_path: Path to 'best.pt' from your training run
    source_path: Path to images, a video, or folder
    """
    # Load your custom trained weights
    model = YOLO(weights_path)
    
    # Run Validation (Testing on the 'test' split defined in your YAML)
    print(f"--- Running Validation/Test ---")
    metrics = model.val() 
    
    # Run Prediction (Visualizing results)
    results = model.predict(source=source_path, save=True, conf=0.25)
    
    return metrics, results

def incremental_train(proj_root, increment=10, model_variant='yolo26n-seg.pt'):
    # Use fixed names so the 'last.pt' is always in the same place
    project_dir = 'mq3_segmentation'
    run_name = 'yolo26_run'
    checkpoint_path = f'runs/segment/{project_dir}/{run_name}/weights/last.pt'
    
    dataset_root = os.path.abspath(proj_root)
    yaml_path = os.path.join(dataset_root, 'data.yaml')

    if os.path.exists(checkpoint_path):
        # 1. Load the model from the checkpoint
        model = YOLO(checkpoint_path)
        
        # 2. Get the index (e.g., if 10 epochs finished, this is 9)
        last_index = model.ckpt.get('epoch', -1) 
        
        # 3. Calculate target: (9 + 1) + 10 = 20
        new_goal = (last_index + 1) + increment
        
        print(f"--- Resuming from Epoch {last_index + 1} ---")
        print(f"--- New Target Goal: {new_goal} ---")
        
        # 4. CRITICAL: Pass resume=True AND the new goal
        model.train(
            data=yaml_path,
            epochs=new_goal,
            resume=True,
            project=project_dir, 
            name=run_name,       
            device=[0, 1],
            batch=32
        )
    else:
        print("--- Starting Fresh Round 1 ---")
        model = YOLO(model_variant)
        model.train(
            data=yaml_path,
            epochs=increment, 
            project=project_dir,
            name=run_name,
            device=[0, 1],
            batch=32,
            mask_ratio=1,
            overlap_mask=True,
            copy_paste=0.3
        )

# YOLOv8 Nano
train_model('yolov8n.pt', 'detection_dataset_split', run_name="yolov8n_crop_run", imgsz=1280, mode="detect")
# test_model('runs/detect/train/weights/best.pt', 'detection_dataset_split/test')

# YOLO26 Nano Segmentation
# incremental_train('segmentation_dataset_split', increment=90, model_variant='yolo26n-seg.pt')
# test_model('runs/segment/mq3_segmentation/yolo26_run/weights/best.pt', 'segmentation_dataset_split/test')