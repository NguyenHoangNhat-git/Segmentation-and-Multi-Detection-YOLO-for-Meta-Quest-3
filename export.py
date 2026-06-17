from ultralytics import YOLO

model = YOLO("custom_semantic_640.pt") 

model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    nms=False,
    simplify=True,
    opset=17,
    end2end=False,
)

with open("custom_sem_labels.txt", "w") as f:
    for name in model.names.values():
        f.write(name+"\n")

model = YOLO("custom_detect_1280.pt") 

model.export(
    format="onnx",
    imgsz=1280,
    nms=False,
    simplify=True,
    opset=17,
)

with open("custom_detect_labels.txt", "w") as f:
    for name in model.names.values():
        f.write(name+"\n")