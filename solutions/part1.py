import cv2
import numpy as np
from ultralytics import YOLO


# Define configuration constants
CONFIDENCE_THRESHOLD_LIMIT = 0.5
BOX_COLOUR = (0, 255, 0)

# Define the device type. Set to "mps" if you want to use M1 Mac GPU. Intel and Nvidia GPU use "gpu"
# Otherwise use "cpu"

DEVICE = "cpu"

# Define video source. You can use a webcam, video file ir a live stream
VIDEO_SOURCE = cv2.VideoCapture(0)  # 0 for webcam

# Load the YOLO model
model = YOLO("yolov8m.pt")

while True:
    # Read the video source
    ret, frame = VIDEO_SOURCE.read()

    # Display the output video
    cv2.imshow("Output video", frame)

    # Stop processing when the "q" key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video source and close the window
VIDEO_SOURCE.release()
cv2.destroyAllWindows()
