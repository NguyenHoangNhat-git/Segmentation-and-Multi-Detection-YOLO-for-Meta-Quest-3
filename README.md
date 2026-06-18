This repo contains the scripts for dataset reparation, model training and model testing for the [Meta Quest Segmentation Pipeline](https://github.com/NguyenHoangNhat-git/Segmentation-Pipeline-on-Meta-Quest-3.git)

## Installation

The original dataset was from [Grasping in the Wild Dataset](https://universe.roboflow.com/iwrist/grasping-in-the-wild), which contains images (+ annotations) of size 1920x1080. To prepare the dataset for training, the following script were used:
- After downloading the 'Grasping in the Wild Dataset', `convert_to_box.py` for turning segmentation mask into bounding box to create a multi-detection dataset.
- `split_dataset.py` for splitting the dataset into `train`, `val` and `test` folder in YOLO format.
- `crop_det_dataset.py` on the detection dataset to create images of size **1280x1280** and `verify_cropped_det_datatset.py` to visualize the results.
- `crop_seg_dataset.py` on the segmentation dataset to create images of size **640x640** and `verify_cropped_seg_datatset.py` to visualize the results.
- `training_testing.py` for training and testing the two models and `export.pt` to convert them into .onnx format with the configuration compatible with the Unity app.