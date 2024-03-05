import cv2  # Import OpenCV library
import numpy as np  # Import NumPy library for numerical operations
import os  # Import os module for interacting with the operating system
import time  # Import time module for timing operations
from datetime import datetime  # Import datetime module for getting current date and time

# Load YOLO model
net = cv2.dnn.readNet("/Users/praveen18kumar/Downloads/yolov4.weights", "/Users/praveen18kumar/Downloads/yolov4.cfg")  # Load YOLO weights and configuration files
classes = []
with open("/Users/praveen18kumar/Downloads/coco.names", "r") as f:  # Open COCO class names file
    classes = [line.strip() for line in f.readlines()]  # Read class names and store in a list
layer_names = net.getLayerNames()  # Get names of all layers in the network
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Get output layer names

# Define input and output folders
input_folder = "/Users/praveen18kumar/Desktop/KeyFrame Exctraction/"  # Input folder path
output_folder = "/Users/praveen18kumar/Desktop/KeyFrame Exctraction/output_images/"  # Output folder path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):  # Check if the output folder does not exist
    os.makedirs(output_folder)  # Create the output folder

# Process images in the input folder
start_time = time.time()  # Record start time of processing
for filename in os.listdir(input_folder):  # Iterate through files in the input folder
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
        filepath = os.path.join(input_folder, filename)  # Get the full path of the image file
        # Load image
        img = cv2.imread(filepath)  # Read the image
        img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Resize image for faster processing
        height, width, channels = img.shape  # Get dimensions of the image

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Create a blob from the image
        net.setInput(blob)  # Set the input for the network
        outs = net.forward(output_layers)  # Forward pass through the network

        # Initialize list to store indices of objects that have been drawn
        drawn_objects = []

        # Showing information on the screen
        for out in outs:  # Iterate through the outputs
            for detection in out:  # Iterate through the detections
                scores = detection[5:]  # Get confidence scores for each class
                class_id = np.argmax(scores)  # Get the class ID with the highest score
                confidence = scores[class_id]  # Get confidence score for the detected class
                if confidence > 0.5:  # Check if confidence is above threshold
                    # Object detected
                    center_x = int(detection[0] * width)  # Calculate center x-coordinate of the bounding box
                    center_y = int(detection[1] * height)  # Calculate center y-coordinate of the bounding box
                    w = int(detection[2] * width)  # Calculate width of the bounding box
                    h = int(detection[3] * height)  # Calculate height of the bounding box

                    x = int(center_x - w / 2)  # Calculate x-coordinate of the top-left corner of the bounding box
                    y = int(center_y - h / 2)  # Calculate y-coordinate of the top-left corner of the bounding box

                    if class_id not in drawn_objects:  # Check if object class has not been drawn yet
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                        cv2.putText(img, classes[class_id], (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)  # Put text label on the bounding box

                        drawn_objects.append(class_id)  # Add object class to the drawn objects list

        # Print current date and time on image
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date and time
        cv2.putText(img, current_datetime, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)  # Put text label with current date and time

        # Save annotated image to output folder
        output_filepath = os.path.join(output_folder, filename)  # Get output file path
        cv2.imwrite(output_filepath, img)  # Save the annotated image

        # Delete original image from input folder
        os.remove(filepath)  # Remove the original image file

print("Processing time:", time.time() - start_time, "seconds")  # Print processing time
