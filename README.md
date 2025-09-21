# HackwithHyderabad - Object Detection for Safety Equipment

## Overview

This project uses YOLOv8 for object detection to identify various types of safety equipment in images. The solution is designed for the HackwithHyderabad hackathon and demonstrates training, validation, and prediction on a custom dataset.

## Dataset Structure

The dataset is provided as zip files and must be extracted before training or testing. There are three main data components:

- `train1.zip`: Contains training images and labels.
- `test1.zip`: Contains validation/test images and labels.
- `scripts.zip`: Contains Python scripts and configuration files, including `classes.txt` and helper scripts.

After extraction, the folder structure should look like this:

```
/content/
├── train1/
│   └── train_1/
│       └── train1/
│           ├── images/
│           └── labels/
├── test1/
│   └── test1/
│       ├── images/
│       └── labels/
├── scripts/
│   └── Hackathon2_scripts/
│       ├── classes.txt
│       ├── train.py
│       ├── predict.py
│       ├── visualize.py
│       ├── yolo_params.yaml
│       └── ENV_SETUP/
```

### Labels

- Labels are provided in YOLO format. Each image has a corresponding `.txt` file in the `labels/` directory.
- Class names are listed in `scripts/Hackathon2_scripts/classes.txt`. The classes are:
  - OxygenTank
  - NitrogenTank
  - FirstAidBox
  - FireAlarm
  - SafetySwitchPanel
  - EmergencyPhone
  - FireExtinguisher

## Requirements

- Python 3.x
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Google Colab (recommended for GPU support)
- Additional dependencies are handled via pip in the notebook/scripts.

## Setup Instructions

1. **Clone the repository and upload the data:**

   Download or clone this repo and upload the `train1.zip`, `test1.zip`, and `scripts.zip` to your Google Drive.

2. **Mount Google Drive in Colab:**

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Copy data from Drive and extract:**

   ```
   # Example in the notebook:
   zip_files_to_copy = ['train1.zip', 'test1.zip', 'scripts.zip']
   for zip_file in zip_files_to_copy:
       !cp "/content/drive/MyDrive/Hackathonhyd/{zip_file}" "/content/{zip_file}"
       !unzip -q "/content/{zip_file}" -d "/content/"
   ```

4. **Install YOLOv8 and dependencies:**

   ```
   %pip install ultralytics
   ```

## Training

- The model is trained using YOLOv8 on the provided dataset.
- The `dataset.yaml` file is automatically generated and points to the correct image and label directories.
- Example training code:

   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8 Nano
   results = model.train(data="dataset.yaml", epochs=10)  # Train for 10 epochs
   ```

- **Note:** Due to limited compute resources, the model was only trained for 10 epochs.  
  **Achieved mAP@.50: 0.534** (mean Average Precision at 0.5 IoU threshold).  
  Training for more epochs or with larger models will likely improve accuracy.

## Testing & Inference

- The best model weights are saved in `runs/detect/train*/weights/best.pt`.
- Run predictions on test images:

   ```python
   results = model.predict(
       source='/content/test1/test1/images',
       conf=0.25,
       iou=0.4,
       save=True,
       save_txt=True,
       project='submission',
       name='preds'
   )
   ```

- Predicted labels are saved in `submission/preds/labels/`.

## Improving Model Performance

- **Current accuracy (mAP@.50): 0.534** after 10 epochs.
- Model performance can be improved by:
  - Increasing the number of epochs (try 50 or more, if resources allow).
  - Using a larger YOLOv8 model (e.g., `yolov8m.pt`, `yolov8l.pt`).
  - Data augmentation and hyperparameter tuning.

## Scripts

- All supporting scripts are located in `scripts/Hackathon2_scripts/`:
  - `train.py`, `predict.py`, `visualize.py`: Main Python scripts for training, prediction, and result visualization.
  - `classes.txt`: List of class names.
  - `yolo_params.yaml`: Example YOLO configuration.

## Acknowledgements

Special thanks to the HackwithHyderabad organizers and contributors of the datasets and tools.

---

For further improvements or to retrain the model, update the `epochs` parameter in the training script and consider experimenting with model size and data augmentation strategies.
