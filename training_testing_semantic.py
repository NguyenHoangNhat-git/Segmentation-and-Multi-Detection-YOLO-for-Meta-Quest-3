import os
from ultralytics import YOLO

def _prepare_dataset(proj_root):
    """
    Safely builds the data.yaml file without triggering syntax errors 
    caused by multiline string indentation or raw paths.
    """
    dataset_root = os.path.abspath(proj_root)
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    
    # We construct the content line by line using explicit double quotes for the path
    yaml_lines = [
        f'path: "{dataset_root}"',
        'train: train/images',
        'val: val/images',
        'test: test/images',
        '',
        'nc: 16',
        "names: ['Bowl', 'CanOfCocaCola', 'FryingPan', 'Glass', 'Jam', 'Lid', 'MilkBottle', 'Mug', 'OilBottle', 'Plate', 'Rice', 'Saucepan', 'Sponge', 'Sugar', 'VinegarBottle', 'WashLiquid']"
    ]
    
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines))
        
    return yaml_path

def train_semantic(proj_root, epochs=100, imgsz=640, run_name="yolo26_semantic_run"):
    """Trains a YOLO26 Semantic Segmentation model directly using polygon text files."""
    yaml_path = _prepare_dataset(proj_root)
    # dataset_root = os.path.abspath(proj_root)
    # yaml_path = os.path.join(dataset_root, 'data.yaml')

    
    # Load the semantic model variant
    model = YOLO("yolo26n-sem.pt")
    
    print("--- Starting Semantic Segmentation Training ---")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,          
        device=[0],
        name=run_name,
        resume=False,
        task="semantic",   # Explicitly set the task to semantic
        
        # --- Optimal Augmentations for Segmentation ---
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

def test_semantic(weights_path, source_path):
    """Validates and predicts using the trained semantic segmentation model."""
    model = YOLO(weights_path)
    
    print("--- Running Semantic Validation ---")
    metrics = model.val() # Evaluates using semantic metrics (mIoU) instead of instance boundaries
    
    print("--- Running Semantic Prediction ---")
    results = model.predict(source=source_path, save=True, conf=0.25)
    return metrics, results

if __name__ == "__main__":
    # 1. Train the semantic model using your existing .txt polygon annotations
    train_semantic(proj_root='cropped_segmentation_dataset_split', epochs=100, run_name="yolo26_semantic_crop_run")
    
    # model = YOLO('runs/semantic/yolo26_semantic400_run/weights/last.pt')
    # results = model.train(
    #     resume=True, 
    # )
    
    # 2. Test the performance of your best semantic weights
    # test_semantic('runs/semantic/yolo26_semantic_run/weights/best.pt', 'segmentation_dataset_split/test')