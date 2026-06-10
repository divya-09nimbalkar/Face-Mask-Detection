# Face Mask Detection

## Overview
Professional face mask detection project with training, evaluation, and live inference.

This implementation uses a MobileNetV2-based classifier and OpenCV face detection to identify whether a person is wearing a mask in images, videos, or webcam streams.

## Features
- Train a custom mask detector from a folder-based dataset
- Evaluate performance with classification reports
- Run inference on images, videos, and live webcam streams
- Annotated output with bounding boxes and confidence scores
- Easy extension for transfer learning and dataset improvements

## Dependencies
This project works best with Python 3.11 or Python 3.12.

Install the required packages:

```bash
pip install -r requirements.txt
```

This project also requires `Pillow` for image loading.

## Dataset Structure
Create a dataset folder with class subdirectories, for example:

```
face_mask_detector/dataset/
  with_mask/
    image_01.jpg
    image_02.jpg
  without_mask/
    image_01.jpg
    image_02.jpg
```

Use exactly `with_mask` and `without_mask` folders for labels.

If the `dataset` folder does not exist, create it inside `face_mask_detector` before training.

The dataset should contain at least two images per label to allow a stratified train/test split.

## Usage

Train a new model:

```bash
python face_mask_detector.py --train dataset --model mask_detector.model
```

Run inference on a single image:

```bash
python face_mask_detector.py --model mask_detector.model --image path/to/image.jpg
```

Run inference on a video file:

```bash
python face_mask_detector.py --model mask_detector.model --video path/to/video.mp4
```

Run live webcam detection:

```bash
python face_mask_detector.py --model mask_detector.model --webcam
```

Press `q` to exit live video mode.

## Notes
- The model is saved as `mask_detector.model` after training.
- The project uses OpenCV Haar cascades for face detection and MobileNetV2 for mask classification.
- For best results, use a balanced dataset with clear face images.

## Author
Divya Nimbalkar
