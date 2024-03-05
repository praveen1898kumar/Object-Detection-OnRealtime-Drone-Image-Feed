# Realtime Drone Object Detection

## Overview
This Python script (`object_detection_realtime.py`) utilizes the YOLO (You Only Look Once) model and OpenCV to perform real-time object detection and annotation on drone image feeds. It draws bounding boxes around detected objects, annotates video frames with the detected object classes, and adds the current date and time. This tool is particularly useful for real-time object detection tasks in drone-based applications.

## Dependencies
- OpenCV (`cv2`): A library for computer vision tasks.
- NumPy (`np`): A library for numerical operations.
- YOLO weights and configuration files.
- COCO class names file.

## How to Use
1. Ensure you have all dependencies installed.
2. Download YOLO weights, configuration files, and COCO class names file.
3. Modify the file paths in the script to match the location of your files.
4. Run the script.

## Explanation of the Code
The code performs the following tasks:
- Loads the YOLO model.
- Reads the COCO class names file.
- Defines input and output folders.
- Processes images in the input folder.
- Detects objects using YOLO.
- Draws bounding boxes around detected objects.
- Annotates images with object classes and current date/time.
- Saves annotated images to the output folder.
- Deletes original images from the input folder after processing.

## Example Usage
```bash
python object_detection_realtime.py
