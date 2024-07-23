import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time

model = YOLO('model/yolov8nChickenV1.pt')

# Capture video from webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/compressed_video.mp4')
cap.set(cv2.CAP_PROP_FPS, 30)

# Define the circle region
circle_center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)
circle_radius = 50  # Radius of the circle
circle_color = (0, 0, 255)  # Red color in BGR
circle_thickness = 2  # Thickness of the circle outline

names = model.model.names

# Tracking history for debugging
track_history = defaultdict(list)

# Initialize FPS calculation
tlast = time.time()
fps_filtered = 30
time.sleep(.1)

while True:
    dt = time.time() - tlast
    fps = 1/dt
    fps_filtered = fps_filtered*.9+fps*.1
    tlast = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    # Detect and track objects
    results = model.track(frame, conf=0.65, iou=0.3, device='cuda', classes=0, persist=True, augment=True, imgsz=320)

    # Draw the circle region
    cv2.circle(frame, circle_center, circle_radius, circle_color, circle_thickness)
    count_inside_circle = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # Track object movement
            track = track_history[track_id]
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=1)

            # Check if the center of the bounding box is inside the circle
            distance_to_center = np.linalg.norm(np.array(bbox_center) - np.array(circle_center))
            if distance_to_center <= circle_radius:
                count_inside_circle += 1
                # Draw bounding box and center point if inside the circle
                #annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.circle(frame, (int(bbox_center[0]), int(bbox_center[1])), 2, (0, 255, 0), -1)

    # Calculate and display FPS
    cv2.rectangle(frame, (0,0),(75,40),(0,0,0),-1)
    cv2.putText(frame, f'FPS: {str(int(fps_filtered))}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Display the count on the frame
    cv2.putText(frame, f'Count: {count_inside_circle}', (5,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    #Resize just for display
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
