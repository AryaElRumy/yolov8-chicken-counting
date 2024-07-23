import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

# Load the exported OpenVINO model
model = YOLO("model/yolov8nChickenV1_openvino_model")

# Open the video file
video_path = "videos/compressed_video.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define the circle region
circle_center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)
circle_radius = 100  # Radius of the circle
circle_color = (0, 0, 255)  # Red color in BGR
circle_thickness = 2  # Thickness of the circle outline

track_history = defaultdict(list)

tlast = time.time()
fps_filtered = 30
time.sleep(.1)

# Loop through the video frames
while cap.isOpened():
    dt = time.time() - tlast
    fps = 1/dt
    fps_filtered = fps_filtered*.9+fps*.1
    tlast = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.track(frame, device='cpu', conf = 0.65, iou = 0.3, classes = 0)

    #draw the circle
    cv2.circle(frame, circle_center, circle_radius, circle_color, circle_thickness)
    count_inside_circle = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids): 
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # Track object movement
            track = track_history[track_id]
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            # Check if the center of the bounding box is inside the circle
            distance_to_center = np.linalg.norm(np.array(bbox_center) - np.array(circle_center))
            if distance_to_center <= circle_radius:
                count_inside_circle += 1
                # Draw center point if inside the circle
                cv2.circle(frame, (int(bbox_center[0]), int(bbox_center[1])), 4, (0, 255, 0), -1)

    # Calculate and display FPS
    cv2.rectangle(frame, (0,0),(75,40),(0,0,0),-1)
    cv2.putText(frame, f'FPS: {str(int(fps_filtered))}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display the count on the frame
    cv2.putText(frame, f'Count: {count_inside_circle}', (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    #Resize just for display
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()