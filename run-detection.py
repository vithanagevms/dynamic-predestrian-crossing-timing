import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Load YOLO model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)  # Tracks objects for up to 30 frames


# video_profile = {
#     "path":"C:/Yoobee/Yoobee-ITS/dynamic-predestrian-crossing-timing/videos/s2/Has_5.mp4",
#     "upper_line" : [(208, 228), (341,227)],
#     "lower_line" : [(282, 327), (561, 271)],
#     #"rotate" : cv2.ROTATE_180,
#     "distance": 10000 #CM unit
# }

# video_profile = {
#     "path":"videos/s2/Han_4.mp4",
#     "upper_line" : [(35, 385), (160, 386)],
#     "lower_line" : [(124, 497), (350, 491)],
#     "rotate" : cv2.ROTATE_90_CLOCKWISE,
#     "distance": 10000 # CM unit
# }

# video_profile = {
#     "path":"videos/s1/Cha_5.mp4",
#     "top_left": (142, 192),
#     "top_right": (263, 193),
#     "bottom_left": (254, 359),
#     "bottom_right": (559, 290),
#     "distance": 10000, # CM unit
#     "videosize": ""
# }

try:
    with open("video_settings.json", "r") as file:
        video_profile = json.load(file)
        video_profile = video_profile[0]
except FileNotFoundError:
    print("Configuration file not found!")
    try:
        with open("CaptureFrameAttributes.py") as captureAttributes:
            exec(captureAttributes.read())
    except FileNotFoundError:
        print("Capture Attributed script missing!")
except json.JSONDecodeError:
    print("Error: Invalid JSON format.")

cap = cv2.VideoCapture(video_profile["video_path"])

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"fps:{fps}, fw:{frame_width}, fh:{frame_height}")
# Define output video writer
##output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Dictionary to track pedestrian data
pedestrian_data = {}

frame_count = 0
frame_skip = 2

# Define pedestrian crossing area (Modify based on your video)
print(video_profile)

crossing_area = [
    video_profile["top_left"],
    video_profile["top_right"],
    video_profile["bottom_right"],
    video_profile["bottom_left"], 
]

# Define exit zones (adjust based on your video)
exit_y_top = (video_profile["top_left"][1] + video_profile["top_right"][1]) //2 # Top of the frame (if moving up)
exit_y_bottom = (video_profile["bottom_left"][1] + video_profile["bottom_right"][1]) //2  # Bottom of the frame (if moving down)
total_distance = exit_y_bottom - exit_y_top

def print_cordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

direction_tracker = {}
pedestrian_times = {}
def update_direction(pedestrian_id, new_center):
    if pedestrian_id in direction_tracker:
        prev_center = direction_tracker[pedestrian_id]
        dx, dy = new_center[0] - prev_center[0], new_center[1] - prev_center[1]
        # if abs(dx) > abs(dy):
        #     direction = "Right" if dx > 0 else "Left"
        # else:
        direction = "Down" if dy > 0 else "Up"
        #print(f"Pedestrian {pedestrian_id} moving: {direction}")
        return direction
    direction_tracker[pedestrian_id] = new_center  # Update stored position


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if(frame_count % frame_skip != 0):
        # print(f"SKIPPED {frame_count}")
        continue
    
    frame = cv2.resize(frame, (640, 360))  # Reduce image size
    if "rotate" in video_profile and video_profile["rotate"] is not None:
        frame = cv2.rotate(frame, video_profile["rotate"])

    # Run YOLO detection
    results = model(frame, classes=[0])
    
    # Draw pedestrian crossing area (Blue transparent overlay)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(crossing_area, np.int32)], (255, 0, 0))  # Blue color
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Transparency effect
    
    cv2.line(frame, (0, exit_y_top), (640, exit_y_top), (125,125,0), 2 )
    cv2.line(frame, (0, exit_y_bottom), (640, exit_y_bottom), (125,125,0), 2 )

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Check if the detected object is a person (COCO class 0)
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2, y2], conf, None))

    # Track pedestrians using DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())

        # Compute center of the person for tracking
        center_x = x2
        center_y = y2
        
        # Track pedestrian entry/exit time
        crossing_time = 0
        if track_id not in pedestrian_times:
            pedestrian_times[track_id] = time.time()
        else:
            crossing_time = time.time() - pedestrian_times[track_id]
            
        speed_px_per_sec = 0
        # Get previous position for speed calculation
        if track_id in pedestrian_data:
            prev_x, prev_y, prev_time = pedestrian_data[track_id]
            distance_px = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            time_diff = time.time() - prev_time

            if time_diff > 0:
                speed_px_per_sec = distance_px / time_diff
                speed_kmh = (speed_px_per_sec * (fps/frame_skip) * 3.6) / 100  # Convert to km/h
            else:
                speed_kmh = 0
        else:
            speed_kmh = 0

        # Predict movement direction (up or down)
        #direction = "Up" if center_y < frame_height // 2 else "Down"
        direction = update_direction(track_id, (center_x, center_y))

        relative_speed = 0
        # Calculate distance to exit
        if direction == "Up":
            distance_to_exit = center_y - exit_y_top
        else:
            distance_to_exit = exit_y_bottom - center_y
            
        if crossing_time > 0: 
            relative_speed = distance_to_exit/crossing_time

        # Store the latest position and timestamp
        pedestrian_data[track_id] = (center_x, center_y, time.time())

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 5, (125, 125,0))

        # Display speed and distance
        cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance_to_exit} px", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {track_id} Dir: {direction}", (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
 
        estimated_time_remaing = 999999
        if speed_px_per_sec > 0:
            estimated_time_remaing = distance_to_exit / speed_px_per_sec
            
        print(f"ID:{track_id}, Speed:{speed_kmh}, Distance:{distance_to_exit}, Dir:{direction}, time: {crossing_time}, Relative Sp:{relative_speed}, Estimated Time remaing: {estimated_time_remaing}")
    # Write frame to output video
    #output_video.write(frame)
    
    # Display for debugging
    cv2.imshow("YOLO + DeepSORT Tracking", frame)
    cv2.setMouseCallback('YOLO + DeepSORT Tracking', print_cordinates)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

cap.release()
#output_video.release()
cv2.destroyAllWindows()
