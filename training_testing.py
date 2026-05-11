from ultralytics import YOLO

def train_model(model_variant, data_yaml, epochs=100, imgsz=640):
    """
    Trains a YOLO model (Detection or Segmentation).
    model_variant: 'yolov8n.pt' or 'yolo26n-seg.pt'
    data_yaml: Path to your dataset config (e.g., 'data.yaml')
    """
    # Load the model
    model = YOLO(model_variant)
    
    # Train
    print(f"--- Starting Training: {model_variant} ---")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=[0, 1],      # Use 'cpu' if you don't have a GPU
        project='grasping_project',
        name='v8n_detection',
        
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
        copy_paste=0.1 # Segment copy-paste (probability)
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


# YOLOv8 Nano
# train_model('yolov8n.pt', 'detection_dataset_split/data.yaml')
# test_model('runs/detect/train/weights/best.pt', 'detection_dataset_split/test')

# YOLO26 Nano Segmentation
train_model('yolo26n-seg.pt', 'segmentation_dataset_split/data.yaml')
test_model('runs/segment/train/weights/best.pt', 'segmentation_dataset_split')