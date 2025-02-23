import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt") #.to("cuda")  # Use GPU for efficiency

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=20)

# Define video input
video_path = "videos/s2/Han_5.mp4"
cap = cv2.VideoCapture(video_path)

# Define pedestrian crossing area (Modify based on your video)
crossing_area = [(282, 327), (561, 271), (341,227), (208, 228)]

# Traffic light settings
red_duration = 10   # Red light duration (seconds)
green_duration = 10  # Green light duration (seconds)

# Initialize traffic light timer
start_time = time.time()
current_state = "RED"  # Start with red light

def update_traffic_light():
    """Update the traffic light status based on elapsed time."""
    global current_state, start_time

    elapsed = time.time() - start_time
    if current_state == "RED" and elapsed >= red_duration:
        current_state = "GREEN"
        start_time = time.time()  # Reset timer
    elif current_state == "GREEN" and elapsed >= green_duration:
        current_state = "RED"
        start_time = time.time()  # Reset timer

def draw_traffic_light_overlay(frame):
    """Draw the traffic light on the video frame."""
    overlay = frame.copy()
    
    # Define light position (top-left corner)
    light_x, light_y = 50, 50
    light_size = 100

    # Draw black background for the traffic light
    cv2.rectangle(overlay, (light_x, light_y), (light_x + light_size, light_y + 150), (0, 0, 0), -1)

    # Determine light color and remaining time
    if current_state == "RED":
        cv2.circle(overlay, (light_x + 50, light_y + 30), 20, (0, 0, 255), -1)  # Red light
        text = "STOP"
    else:
        cv2.circle(overlay, (light_x + 50, light_y + 120), 20, (0, 255, 0), -1)  # Green light
        text = "GO"

    remaining_time = int((red_duration if current_state == "RED" else green_duration) - (time.time() - start_time))
    
    # Display status text and countdown timer
    cv2.putText(overlay, text, (light_x + 10, light_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"{remaining_time}s", (light_x + 15, light_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return overlay

def is_inside_crossing(x1, y1, x2, y2, area_points):
    """Check if the center of a bounding box is inside the pedestrian crossing area."""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(np.array(area_points, np.int32), (center_x, center_y), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update traffic light state
    update_traffic_light()
    
    frame = cv2.resize(frame, (640, 360))  # Reduce image size
    frame = cv2.rotate(frame, cv2.ROTATE_180);

    # Draw pedestrian crossing area (Blue transparent overlay)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(crossing_area, np.int32)], (255, 0, 0))  # Blue color for crossing
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Transparency effect

    # Run YOLO detection
    results = model(frame)

    # Store detections for tracking
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Only detect pedestrians (COCO class 0)
            if cls == 0 and conf > 0.5 and is_inside_crossing(x1, y1, x2, y2, crossing_area):
                detections.append(([x1, y1, x2, y2], conf, None))

    # Track objects with DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())

        # Check if the pedestrian is inside the crossing area
        if is_inside_crossing(x1, y1, x2, y2, crossing_area):
            if current_state == "RED":
                # If red light, draw bounding box in RED
                color = (0, 0, 255)
                cv2.putText(frame, f"ID: {track_id} (VIOLATION)", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # If green light, draw bounding box in BLUE
                color = (255, 0, 0)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Add traffic light to the frame
    frame = draw_traffic_light_overlay(frame)

    # Show frame
    cv2.imshow("Pedestrian Tracking with Traffic Light", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()