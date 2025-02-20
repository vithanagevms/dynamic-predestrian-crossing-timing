import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for a more accurate model

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)  # Tracks objects for up to 30 frames

# Load video
video_path = "v1.mp4"
cap = cv2.VideoCapture(video_path)


# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps, frame_width, frame_height)

# Define output video writer
output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Dictionary to track pedestrian crossing times
pedestrian_times = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Check if the detected object is a person (COCO class 0)
            if cls == 0 and conf > 0.4:
                detections.append(([x1, y1, x2, y2], conf, None))

    # Track pedestrians using DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())

        # Draw tracking info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.putText(frame, f"SPEED: {track.speed}")

        # Track pedestrian entry/exit time
        if track_id not in pedestrian_times:
            pedestrian_times[track_id] = time.time()
        else:
            crossing_time = time.time() - pedestrian_times[track_id]
            cv2.putText(frame, f"Time: {crossing_time:.1f}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Write frame to output video
    output_video.write(frame)
    
    # Display for debugging
    cv2.imshow("YOLO + DeepSORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()