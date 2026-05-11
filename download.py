from ultralytics import YOLO

def download_models():
    print("Initializing downloads...")

    # 1. Download YOLOv8 Nano (Detection)
    print("\n--- Downloading YOLOv8n ---")
    model_v8 = YOLO('yolov8n.pt') 
    
    # 2. Download YOLO26 Nano (Segmentation)
    # YOLO26 is the 2026 flagship model optimized for NMS-free inference
    print("\n--- Downloading YOLO26n-seg ---")
    model_26 = YOLO('yolo26n-seg.pt')

    print("\nDownloads complete! Weights saved in your current directory.")

if __name__ == "__main__":
    download_models()