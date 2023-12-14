By: Zaidan Muhammad Rafi (Tugas UAS Kecerdasan Buatan)

OBJECT DETECTION ON IMAGE AND VIDEO USING YOLO V8 ON GOOGLE COLAB

Colab: https://colab.research.google.com/drive/17ANVVp8UssQN2OVHzgGGKplck1QLogVe?usp=sharing

**---------------------------------------------------------------------------------------------------------**

Explanation:

The program aims to carry out object detection using the YOLOv8 model on the Google Colab platform. After importing the necessary libraries and installing Ultralytics, the program loads the YOLOv8 model. The perform_object_detection_on_frame function is then created to perform object detection in an image frame. Users can upload image or video files, and the program automatically detects objects in the image or each video frame. Object detection results are displayed in the form of images with bounding boxes, and if the uploaded file is a video, the program will produce an output video with the detected objects. This program combines the advantages of Google Colab in providing computing resources and Ultralytics to leverage the YOLOv8 model, providing users with an easy and practical object detection experience.


Code Detail Explanation:
	
**1.	Import Library dan Package Initialization:**
```
from google.colab import files
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
import numpy as np

# Install Ultralytics
!pip install ultralytics
```

**2.	Import YOLO from Ultralytics:**
```
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8m.pt")
```

**3.	‘perform_object_detection_on_frame’ Function:**
```
def perform_object_detection_on_frame(frame):
    # Perform object detection
    results = model.predict(frame, conf=0.35)
    result = results[0]
    
    # Convert the result to an OpenCV format
    result_cv2 = np.array(result.plot()[:, :, ::-1])

    return result_cv2
```

**4.	Inputted File Processing:**
```
# Allow file upload
uploaded = files.upload()

# Process each uploaded file
for file_name, content in uploaded.items():
```

**5.	Object Detection on Image:**
```
if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Convert the content to an image
        image = Image.open(BytesIO(content))

        # Perform object detection on the uploaded image
        result_frame = perform_object_detection_on_frame(np.array(image))

        # Display the image with bounding boxes
        image_with_boxes = Image.fromarray(result_frame)
        display(image_with_boxes)
```

**6.	Object Detection on Video:**
```
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
  out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)) # Adjust parameters as needed

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
```

**---------------------------------------------------------------------------------------------------------**

**Output on Image:**
1. Before
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/df3803a2-65f8-4b43-806d-c385f5734f16)
After:
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/ccdf1b4d-bd96-4ac1-aebd-0aa78f219a29)

2. Before
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/e87d6f79-08fe-4426-bee7-b0134aee9474)
After:
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/be7b8716-334c-44c0-8afb-71e330448dda)

3. Before
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/cc18be2f-c959-4753-8e9c-f98e1fe8ccbf)
After:
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/8359ce01-4d18-4ab1-b536-b3c3b5c54a4d)

4. Before
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/9b808cea-078d-462a-a845-6b84fb2ba52a)
After:
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/cfa64b19-553e-44c6-b766-6648e8c466fb)

5. Before
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/d6a2d1ec-d2ec-46b9-b097-47799038881c)
After:
![image](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/5e6c1114-d6fe-4a5f-b601-c6235f4029f8)


**Output on VIdeo:**

Before:


![video_sebelum_yolov8](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/b0e84225-b38d-4c12-8c49-60f2fe59abed)



After:

![video_sesudah_yolov8](https://github.com/zaidanrafi/Object-detection-on-image-and-video-using-YOLO-v8/assets/41849571/ae7de3bf-c65d-489a-9f41-47848c95d5ad)


**-----------------**
