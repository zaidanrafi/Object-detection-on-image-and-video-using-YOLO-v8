from google.colab import files
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
import numpy as np

# Install Ultralytics
!pip install ultralytics

# Import YOLO from Ultralytics
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8m.pt")

def perform_object_detection_on_frame(frame):
    # Perform object detection
    results = model.predict(frame, conf=0.35)
    result = results[0]

    # Convert the result to an OpenCV format
    result_cv2 = np.array(result.plot()[:, :, ::-1])

    return result_cv2

# Allow file upload
uploaded = files.upload()

# Process each uploaded file
for file_name, content in uploaded.items():
    # Check if the file is an image or a video
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Convert the content to an image
        image = Image.open(BytesIO(content))

        # Perform object detection on the uploaded image
        result_frame = perform_object_detection_on_frame(np.array(image))

        # Display the image with bounding boxes
        image_with_boxes = Image.fromarray(result_frame)
        display(image_with_boxes)

    elif file_name.lower().endswith(('.mp4', '.avi', '.mkv')):
        # Convert the content to a video file
        video_path = f"/content/{file_name}"
        with open(video_path, 'wb') as f:
            f.write(content)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Create VideoWriter for the output video
        output_path = f"/content/output_{file_name}"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))  # Adjust parameters as needed

        # Process each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection on the frame
            result_frame = perform_object_detection_on_frame(frame)

            # Write the frame to the output video
            out.write(result_frame)

            # Display the frame with bounding boxes
            frame_with_boxes = Image.fromarray(result_frame)
            display(frame_with_boxes, display_id='frame_with_boxes')

        # Release resources
        cap.release()
        out.release()

        # Display the output video for download
        files.download(output_path)
