import cv2
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PedestrianTracker:
    """
    A class to track pedestrians using YOLO for detection and DeepSORT for tracking.
    Implements OOP principles for better code organization and maintainability.
    """
    
    def __init__(self, config_path: str = "video_settings.json", yolo_model: str = "yolov8n.pt", frame_skip: int = 5):
        """
        Initialize the PedestrianTracker with configuration settings.
        
        Args:
            config_path: Path to the JSON configuration file
            yolo_model: YOLO model to use for detection
            frame_skip: Number of frames to skip between processing
        """
        # Configure logging
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Hide YOLO logs
        
        # Initialize tracking parameters
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.pedestrian_data = {}
        self.direction_tracker = {}
        self.pedestrian_times = {}
        self.pedestrian_way_up = []
        self.pedestrian_way_down = []
        
        # Estimation system parameters
        self.zone_first_entry_time = None
        self.estimation_triggered = False
        self.estimation_delay = 5  # seconds before checking slowest pedestrian
        self.current_pedestrians_in_zone = set()
        self.completed_pedestrians = set()
        self.pedestrian_estimates = {}
        
        # Load models
        self.detection_model = YOLO(yolo_model)
        self.tracker = DeepSort(max_age=10)  # Tracks objects for up to 10 frames
        
        # Load configuration
        self.video_profile = self._load_config(config_path)
        
        # Setup video capture
        self.cap = self._setup_video_capture()
        
        # Define tracking zones based on configuration
        self._setup_tracking_zones()
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as file:
                video_profile = json.load(file)
                return video_profile[0]
        except FileNotFoundError:
            print(f"Configuration file not found at {config_path}!")
            try:
                with open("CaptureFrameAttributes.py") as captureAttributes:
                    exec(captureAttributes.read())
                    # This would need proper handling in a production environment
            except FileNotFoundError:
                print("Capture Attributes script missing!")
                raise
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            raise
    
    def _setup_video_capture(self) -> cv2.VideoCapture:
        """
        Set up the video capture from the configured source.
        
        Returns:
            OpenCV VideoCapture object
        """
        try:
            cap = cv2.VideoCapture(self.video_profile["path"])
            if not cap.isOpened():
                raise FileNotFoundError(f"Failed to open video file at {self.video_profile['path']}")
                
            # Get video properties
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video properties - FPS: {self.fps}, Width: {self.frame_width}, Height: {self.frame_height}")
            
            return cap
        except Exception as e:
            print(f"Error setting up video capture: {e}")
            raise
    
    def _setup_tracking_zones(self) -> None:
        """
        Define the pedestrian crossing area and exit zones based on configuration.
        """
        # Define pedestrian crossing area
        self.crossing_area = [
            self.video_profile["top_left"],
            self.video_profile["top_right"],
            self.video_profile["bottom_right"],
            self.video_profile["bottom_left"], 
        ]
        
        # Define exit zones
        self.exit_y_top = (self.video_profile["top_left"][1] + self.video_profile["top_right"][1]) // 2
        self.exit_y_bottom = (self.video_profile["bottom_left"][1] + self.video_profile["bottom_right"][1]) // 2
        self.total_distance = self.exit_y_bottom - self.exit_y_top
        
        # Create a numpy array of the crossing area for point-in-polygon testing
        self.crossing_area_np = np.array(self.crossing_area, dtype=np.int32)
    
    def measure_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Measure the Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def update_direction(self, pedestrian_id: int, new_center: Tuple[int, int]) -> Optional[str]:
        """
        Update and determine the movement direction of a pedestrian.
        
        Args:
            pedestrian_id: Unique identifier for the pedestrian
            new_center: New position (x, y)
            
        Returns:
            Direction of movement ("Up" or "Down")
        """
        if pedestrian_id in self.direction_tracker:
            prev_center = self.direction_tracker[pedestrian_id]
            _, dy = new_center[0] - prev_center[0], new_center[1] - prev_center[1]
            direction = "Down" if dy > 0 else "Up"
            self.direction_tracker[pedestrian_id] = new_center  # Update position
            return direction
            
        self.direction_tracker[pedestrian_id] = new_center  # Store initial position
        return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for pedestrian detection and tracking.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with visualization
        """
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 360))
        
        # Apply rotation if specified in configuration
        if "rotate" in self.video_profile:
            if self.video_profile["rotate"] == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.video_profile["rotate"] == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.video_profile["rotate"] == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Run detection (only people class for optimum resource allocation)
        results = self.detection_model(frame, classes=[0])
        
        # Visualize crossing area and exit zones
        frame = self._visualize_zones(frame)
        
        # Extract detections from YOLO results
        detections = self._extract_detections(results)
        
        # Track pedestrians using DeepSORT
        tracked_objects = self.tracker.update_tracks(detections, frame=frame)
        
        # Process and visualize each tracked pedestrian
        frame = self._process_tracks(frame, tracked_objects)
        
        return frame
    
    def _update_estimation_system(self, current_time: float, active_pedestrian_ids: set) -> None:
        """
        Update the estimation system state and trigger estimations when needed.
        
        Args:
            current_time: Current timestamp
            active_pedestrian_ids: Set of currently active pedestrian IDs
        """
        # Skip if first entry hasn't happened yet
        if self.zone_first_entry_time is None:
            return
            
        # Check for disappeared pedestrians (no longer tracked)
        disappeared_pedestrians = set(self.current_pedestrians_in_zone) - active_pedestrian_ids
        for p_id in disappeared_pedestrians:
            self.current_pedestrians_in_zone.remove(p_id)
            print(f"Pedestrian {p_id} disappeared from tracking")
            
        # Check if it's time to trigger the estimation after delay
        if not self.estimation_triggered and (current_time - self.zone_first_entry_time) >= self.estimation_delay:
            self.estimation_triggered = True
            slowest_id, completion_time = self._estimate_slowest_pedestrian_completion()
            
            print(f"\n===== ESTIMATION AFTER {self.estimation_delay} SECONDS =====")
            print(f"Time since first pedestrian entered: {current_time - self.zone_first_entry_time:.2f} seconds")
            print(f"Number of pedestrians in zone: {len(self.current_pedestrians_in_zone)}")
            
            if slowest_id is not None:
                print(f"Slowest pedestrian ID: {slowest_id}")
                print(f"Estimated completion time: {completion_time:.2f} seconds")
                print(f"Estimated completion at: {time.ctime(current_time + completion_time)}")
            else:
                print("No valid estimations available yet")
            print("=======================================\n")
    
    def _estimate_slowest_pedestrian_completion(self) -> Tuple[Optional[int], float]:
        """
        Estimate which pedestrian will take the longest to complete crossing the zone.
        
        Returns:
            Tuple of (slowest_pedestrian_id, estimated_completion_time)
        """
        slowest_id = None
        longest_time = -1
        
        # Check all pedestrians currently in the zone
        for p_id in self.current_pedestrians_in_zone:
            if p_id in self.pedestrian_estimates:
                est = self.pedestrian_estimates[p_id]
                if est['estimated_time'] > longest_time:
                    longest_time = est['estimated_time']
                    slowest_id = p_id
                    
        return slowest_id, longest_time
    
    def _visualize_estimations(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize estimation information on the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with estimation visualization
        """
        # Add estimation information to the top of the frame
        if self.zone_first_entry_time is not None:
            cv2.putText(frame, f"Zone Active: {len(self.current_pedestrians_in_zone)} pedestrians", 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show estimation after trigger
            if self.estimation_triggered:
                slowest_id, completion_time = self._estimate_slowest_pedestrian_completion()
                if slowest_id is not None:
                    cv2.putText(frame, f"Slowest: ID {slowest_id} - Est. {completion_time:.1f}s", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        return frame
    
    def _visualize_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize crossing area and exit zones on the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with visualization overlays
        """
        # Draw pedestrian crossing area (Blue transparent overlay)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(self.crossing_area, np.int32)], (255, 0, 0))  # Blue color
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Transparency effect
        
        # Draw exit lines
        cv2.line(frame, (0, self.exit_y_top), (640, self.exit_y_top), (125, 125, 0), 2)
        cv2.line(frame, (0, self.exit_y_bottom), (640, self.exit_y_bottom), (125, 125, 0), 2)
        
        return frame
    
    def _extract_detections(self, results) -> List:
        """
        Extract detections from YOLO results for DeepSORT tracking.
        
        Args:
            results: YOLO detection results
            
        Returns:
            List of detections in DeepSORT format
        """
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Check if the detected object is a person (COCO class 0) with high confidence
                if cls == 0 and conf > 0.5:
                    detections.append(([x1, y1, x2, y2], conf, None))
        
        return detections
    
    def _process_tracks(self, frame: np.ndarray, tracked_objects) -> np.ndarray:
        """
        Process each tracked pedestrian and visualize metrics.
        
        Args:
            frame: Input video frame
            tracked_objects: List of tracked objects from DeepSORT
            
        Returns:
            Frame with visualized tracking information
        """
        current_time = time.time()
        active_pedestrian_ids = set()
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            active_pedestrian_ids.add(track_id)
            ltwh = track.to_ltwh()
            x1, y1, x2, y2 = map(int, ltwh)
            #x2, y2 = x1 + w, y1 + h
            
            # Compute bottom center of the person for tracking
            object_bottom_x = (x1 + x2) // 2
            object_bottom_y = y2
            
            # Check if pedestrian is in the crossing zone
            in_zone = self._check_in_zone(object_bottom_x, object_bottom_y)
            
            # Calculate metrics for the tracked pedestrian
            metrics = self._calculate_pedestrian_metrics(track_id, object_bottom_x, object_bottom_y, current_time, in_zone)
            
            # Store the latest position and timestamp
            self.pedestrian_data[track_id] = (object_bottom_x, object_bottom_y, current_time)
            
            # Visualize the pedestrian and metrics
            frame = self._visualize_pedestrian(frame, track_id, x1, y1, x2, y2, metrics)
        
        # Update estimation system
        self._update_estimation_system(current_time, active_pedestrian_ids)
        
        # Visualize estimation information
        frame = self._visualize_estimations(frame)
        
        return frame
    
    def _check_in_zone(self, x: int, y: int) -> bool:
        """
        Check if a point is inside the crossing zone using point-in-polygon test.
        
        Args:
            x: X-coordinate of the point
            y: Y-coordinate of the point
            
        Returns:
            True if point is in the crossing zone, False otherwise
        """
        point = (x, y)
        return cv2.pointPolygonTest(self.crossing_area_np, point, False) >= 0
    
    def _calculate_pedestrian_metrics(self, track_id: int, x: int, y: int, current_time: float, in_zone: bool) -> dict:
        """
        Calculate metrics for a tracked pedestrian.
        
        Args:
            track_id: Unique identifier for the pedestrian
            x: X-coordinate of the pedestrian
            y: Y-coordinate of the pedestrian
            current_time: Current timestamp
            in_zone: Whether the pedestrian is in the crossing zone
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'direction': None,
            'crossing_time': 0,
            'speed_kmh': 0,
            'distance_to_exit': 0,
            'relative_speed': 0,
            'estimated_time_remaining': 999999,
            'in_zone': in_zone
        }
        
        # Track pedestrian entry/exit time
        if track_id not in self.pedestrian_times:
            self.pedestrian_times[track_id] = current_time
        else:
            metrics['crossing_time'] = current_time - self.pedestrian_times[track_id]
        
        # Predict movement direction (up or down)
        metrics['direction'] = self.update_direction(track_id, (x, y))
        
        # Track zone entry - important for the estimation system
        if in_zone:
            if self.zone_first_entry_time is None:
                self.zone_first_entry_time = current_time
                print(f"First pedestrian (ID: {track_id}) entered the zone at time {current_time}")
            
            # Add pedestrian to the current set in zone
            self.current_pedestrians_in_zone.add(track_id)
        elif track_id in self.current_pedestrians_in_zone:
            # Pedestrian has left the zone
            self.current_pedestrians_in_zone.remove(track_id)
            self.completed_pedestrians.add(track_id)
            print(f"Pedestrian (ID: {track_id}) has left the zone at time {current_time}")
        
        # Get previous position for speed calculation
        if track_id in self.pedestrian_data:
            prev_x, prev_y, prev_time = self.pedestrian_data[track_id]
            distance_px = self.measure_distance((x, y), (prev_x, prev_y))
            time_diff = current_time - prev_time
            
            if time_diff > 0:
                speed_px_per_sec = distance_px / time_diff
                metrics['speed_kmh'] = (speed_px_per_sec * (self.fps/self.frame_skip) * 3.6) / 100  # Convert to km/h
                
                # Calculate distance to exit
                if metrics['direction'] == "Up":
                    metrics['distance_to_exit'] = y - self.exit_y_top
                else:
                    metrics['distance_to_exit'] = self.exit_y_bottom - y
                
                if metrics['crossing_time'] > 0:
                    metrics['relative_speed'] = metrics['distance_to_exit'] / metrics['crossing_time']
                
                if speed_px_per_sec > 0:
                    metrics['estimated_time_remaining'] = metrics['distance_to_exit'] / speed_px_per_sec
                    
                    # Store the estimation for this pedestrian
                    self.pedestrian_estimates[track_id] = {
                        'speed': speed_px_per_sec,
                        'distance': metrics['distance_to_exit'],
                        'estimated_time': metrics['estimated_time_remaining'],
                        'last_update': current_time
                    }
        
        return metrics
    
    def _visualize_pedestrian(self, frame: np.ndarray, track_id: int, x1: int, y1: int, x2: int, y2: int, metrics: dict) -> np.ndarray:
        """
        Visualize a tracked pedestrian and their metrics on the frame.
        
        Args:
            frame: Input video frame
            track_id: Unique identifier for the pedestrian
            x1, y1, x2, y2: Bounding box coordinates
            metrics: Dictionary of calculated metrics
            
        Returns:
            Frame with visualization
        """
        # Determine box color based on whether pedestrian is in the zone
        box_color = (0, 255, 0)  # Default green
        if metrics['in_zone']:
            box_color = (0, 0, 255)  # Red for pedestrians in crossing zone
            
        # Check if this is the estimated slowest pedestrian
        if self.estimation_triggered:
            slowest_id, _ = self._estimate_slowest_pedestrian_completion()
            if track_id == slowest_id:
                box_color = (0, 165, 255)  # Orange for slowest pedestrian
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 5, (125, 125, 0))
        
        # Display metrics
        cv2.putText(frame, f"Speed: {metrics['speed_kmh']:.1f} km/h", (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Distance: {metrics['distance_to_exit']} px", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, f"ID: {track_id} Dir: {metrics['direction']}", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # For pedestrians in zone, show estimated remaining time
        if metrics['in_zone'] and metrics['estimated_time_remaining'] < 999999:
            cv2.putText(frame, f"Est: {metrics['estimated_time_remaining']:.1f}s", (x1, y2 + 65), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # Print detailed metrics to console (could be logged to a file instead)
        if metrics['direction']:
            print(f"ID:{track_id}, Speed:{metrics['speed_kmh']:.1f}, Distance:{metrics['distance_to_exit']}, " 
                  f"Dir:{metrics['direction']}, Time: {metrics['crossing_time']:.2f}, "
                  f"In Zone: {metrics['in_zone']}, "
                  f"Estimated Time Remaining: {metrics['estimated_time_remaining']:.2f}")
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for interactive coordinate display.
        
        Args:
            event: Mouse event type
            x, y: Cursor coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Coordinates: ({x}, {y})")
    
    def run(self):
        """
        Main method to run the pedestrian tracking system.
        """
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Process the current frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Pedestrian Tracking System", processed_frame)
                cv2.setMouseCallback('Pedestrian Tracking System', self.mouse_callback)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)  # Wait until any key is pressed (pause)
                
        finally:
            # Clean up resources
            self.cap.release()
            cv2.destroyAllWindows()
            print("Pedestrian tracking completed.")


class PedestrianAnalytics:
    """
    A class to analyze pedestrian data collected by the PedestrianTracker.
    This could be extended with methods to generate reports, heatmaps, etc.
    """
    
    def __init__(self):
        """Initialize the analytics system."""
        self.pedestrian_records = {}
    
    def add_record(self, pedestrian_id, timestamp, position, speed, direction):
        """Add a pedestrian tracking record to the analytics system."""
        if pedestrian_id not in self.pedestrian_records:
            self.pedestrian_records[pedestrian_id] = []
        
        self.pedestrian_records[pedestrian_id].append({
            'timestamp': timestamp,
            'position': position,
            'speed': speed,
            'direction': direction
        })
    
    def generate_summary(self):
        """Generate a summary of pedestrian analytics."""
        total_pedestrians = len(self.pedestrian_records)
        direction_up = sum(1 for p_id in self.pedestrian_records 
                         if self.pedestrian_records[p_id][-1]['direction'] == 'Up')
        direction_down = total_pedestrians - direction_up
        
        avg_speeds = []
        for p_id, records in self.pedestrian_records.items():
            speeds = [r['speed'] for r in records if r['speed'] > 0]
            if speeds:
                avg_speeds.append(sum(speeds) / len(speeds))
        
        avg_speed = sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0
        
        return {
            'total_pedestrians': total_pedestrians,
            'direction_up': direction_up,
            'direction_down': direction_down,
            'average_speed': avg_speed
        }


def main():
    """Main function to run the pedestrian tracking system."""
    try:
        # Create and run the pedestrian tracker
        tracker = PedestrianTracker(
            config_path="video_settings.json",
            yolo_model="yolov8n.pt",
            frame_skip=5
        )
        
        # Set custom parameters for the estimation system if needed
        tracker.estimation_delay = 5  # seconds after first entry to trigger estimation
        
        # Run the tracker
        tracker.run()
        
    except Exception as e:
        print(f"Error running pedestrian tracker: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()