import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import numpy as np
import mysql.connector
import os

# MySQL database connection parameters
db_config = {
    'user': 'root',
    'host': "localhost",
    'database': 'object_detection'
}

# Function to create the database if it doesn't exist
def create_database(cursor):
    cursor.execute("CREATE DATABASE IF NOT EXISTS object_detection")

# Function to create the table if it doesn't exist
def create_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        count INT,
        frame_path VARCHAR(255),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

# Connect to MySQL server
db = mysql.connector.connect(**db_config)
cursor = db.cursor()

# Create the database if it doesn't exist
create_database(cursor)

# Select the database
cursor.execute("USE object_detection")

# Create the table if it doesn't exist
create_table(cursor)

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

# Create a directory to store the frames
frames_dir = "captured_frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

frame_counter = 0

# Loop through the video frames
while cap.isOpened():
    dt = time.time() - tlast
    fps = 1 / dt
    fps_filtered = fps_filtered * .9 + fps * .1
    tlast = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.track(frame, device='cpu', conf=0.65, iou=0.3, classes=0)

    # Draw the circle
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

    if count_inside_circle > 0:
        # Save the frame as a JPEG file
        frame_filename = f"frame_{frame_counter}_count_{count_inside_circle}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_counter += 1

        # Log count and frame path to MySQL
        cursor.execute("INSERT INTO counts (count, frame_path) VALUES (%s, %s)", (count_inside_circle, frame_path))
        db.commit()

    # Calculate and display FPS
    cv2.rectangle(frame, (0, 0), (75, 40), (0, 0, 0), -1)
    cv2.putText(frame, f'FPS: {str(int(fps_filtered))}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display the count on the frame
    cv2.putText(frame, f'Count: {count_inside_circle}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Resize just for display
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
