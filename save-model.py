import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import json
import time
import yaml


class PedestrianCrossingDetector:
    def __init__(self, crossing_config):
        """
        Initialize the detector with crossing configuration
        crossing_config: dict with 'crossing_length' in meters and 'crossing_points' coordinates
        """
        self.crossing_config = crossing_config
        self.model = None
        self.tracker = DeepSort(max_age=30)
        self.pedestrian_tracks = {}
        self.model_path = None

    def train_model(self, train_data_path, save_dir='./models'):
        """
        Train custom YOLO model on pedestrian crossing data and save it
        """
        os.makedirs(save_dir, exist_ok=True)

        # Initialize YOLO model with custom configuration
        self.model = YOLO('yolov8n.yaml')

        abs_data_path = os.path.abspath(train_data_path)
        abs_images_train = os.path.join(abs_data_path, 'images', 'train')
        abs_images_val = os.path.join(abs_data_path, 'images', 'val')

        data_yaml_path = os.path.join(abs_data_path, 'data.yaml')

        # Create data.yaml file for training
        data_yaml = {
            'path': abs_data_path,
            'train': abs_images_train,
            'val': abs_images_val,
            'nc': 1,
            'names': ['pedestrian']
        }

        with open(data_yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

        # Start training with save directory
        self.model.train(
            data=data_yaml_path,
            epochs=2,
            batch=16,
            imgsz=640,
            project=save_dir,
            name='pedestrian_detector'
        )

        # Save the model path
        self.model_path = os.path.join(save_dir, 'pedestrian_detector', 'weights', 'best.pt')
        print(f"Model saved to: {self.model_path}")
        return self.model_path

    def load_model(self, model_path=None):
        """
        Load a previously trained model
        """
        if model_path is None and self.model_path is not None:
            model_path = self.model_path
        elif model_path is None:
            raise ValueError("No model path provided and no previously saved model found")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def calculate_speed(self, positions, fps):
        """
        Calculate speed from positions history
        Returns speed in meters per second
        """
        if len(positions) < 2:
            return 0

        # Calculate pixel distance between last two positions
        p1 = np.array(positions[-2])
        p2 = np.array(positions[-1])
        pixel_distance = np.linalg.norm(p2 - p1)

        # Convert to real-world distance using crossing length
        real_distance = (pixel_distance * self.crossing_config['crossing_length'] /
                         self.crossing_config['crossing_width_pixels'])

        # Calculate speed (distance / time)
        time_diff = 1 / fps
        speed = real_distance / time_diff

        return speed

    def process_video(self, video_path, output_path):
        """
        Process video and track pedestrians
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model(frame)

            # Process detections
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    if conf > 0.4:  # Confidence threshold
                        detections.append(([x1, y1, x2, y2], conf, cls))

            # Update tracks
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr()

                # Update position history
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {
                        'positions': [],
                        'entry_time': time.time(),
                        'crossed': False
                    }

                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.pedestrian_tracks[track_id]['positions'].append(center)

                # Calculate speed
                speed = self.calculate_speed(
                    self.pedestrian_tracks[track_id]['positions'],
                    fps
                )

                # Check if pedestrian has crossed
                if self.check_crossing_completion(center):
                    if not self.pedestrian_tracks[track_id]['crossed']:
                        self.pedestrian_tracks[track_id]['crossed'] = True
                        self.pedestrian_tracks[track_id]['crossing_time'] = (
                                time.time() - self.pedestrian_tracks[track_id]['entry_time']
                        )
                        print("Crossing Time", self.pedestrian_tracks[track_id]['crossing_time'])

                # Draw tracking info
                self.draw_tracking_info(
                    frame,
                    bbox,
                    track_id,
                    speed,
                    self.pedestrian_tracks[track_id].get('crossing_time', None)
                )

            # Draw crossing area
            self.draw_crossing_area(frame)

            # Write frame
            output_video.write(frame)

        cap.release()
        output_video.release()

    def check_crossing_completion(self, position):
        """
        Check if a position has completed crossing
        """
        # Implementation depends on crossing_config geometry
        x, y = position
        crossing_end = self.crossing_config['crossing_points'][1]
        return y > crossing_end[1]

    def draw_tracking_info(self, frame, bbox, track_id, speed, crossing_time):
        """
        Draw tracking information on frame
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ID and speed
        cv2.putText(
            frame,
            f"ID: {track_id} Speed: {speed:.1f} m/s",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Draw crossing time if available
        if crossing_time is not None:
            cv2.putText(
                frame,
                f"Crossing Time: {crossing_time:.1f}s",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    def draw_crossing_area(self, frame):
        """
        Draw crossing area on frame
        """
        points = np.array(self.crossing_config['crossing_points'], np.int32)
        cv2.polylines(frame, [points], True, (255, 255, 0), 2)


def prepare_dataset(video_paths, output_dir):
    """
    Prepare dataset from videos with automatic labeling using pre-trained YOLO
    """
    # Initialize pre-trained YOLO model for generating initial labels
    pretrained_model = YOLO('yolov8n.pt')

    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)

    # Create directories for images and labels
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')

    # Create all directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    frame_count = 0
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue

        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Decide if frame goes to train or val (80/20 split)
            subset = 'train' if np.random.rand() < 0.8 else 'val'

            # Create frame filename
            frame_filename = f'{video_name}_frame_{frame_count:06d}.jpg'
            label_filename = f'{video_name}_frame_{frame_count:06d}.txt'

            # Save frame
            frame_path = os.path.join(output_dir, 'images', subset, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Generate labels using pre-trained model
            results = pretrained_model(frame)

            detection_count = 0
            # Create label file
            label_path = os.path.join(output_dir, 'labels', subset, label_filename)
            with open(label_path, 'w') as f:
                # For each detected person (class 0 in COCO)
                for r in results:
                    for box, cls in zip(r.boxes.xywhn, r.boxes.cls):
                        if int(cls) == 0:  # If detection is a person
                            detection_count += 1
                            # Convert box to YOLO format (class x_center y_center width height)
                            x_center, y_center, width, height = box.tolist()
                            # Write in YOLO format
                            f.write(f"0 {x_center} {y_center} {width} {height}\n")

            if detection_count > 0:
                print(f"Found {detection_count} people in frame {frame_count}")

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()

    print(f"Processed {frame_count} frames across {len(video_paths)} videos")
    return True


def main():
    # Configuration
    crossing_config = {
        'crossing_length': 10,  # meters
        'crossing_width_pixels': 500,  # pixels
        'crossing_points': [
            [100, 100],  # top-left
            [600, 100],  # top-right
            [600, 400],  # bottom-right
            [100, 400]  # bottom-left
        ]
    }

    # Initialize detector
    detector = PedestrianCrossingDetector(crossing_config)

    # Set paths
    dataset_path = os.path.abspath('./prepared_dataset')
    model_save_path = './models/pedestrian_detector/weights/best.pt'

    # Check if model exists
    if os.path.exists(model_save_path):
        print("Loading existing model...")
        try:
            detector.load_model(model_save_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Training new model...")
        # Prepare dataset
        train_videos = [os.path.abspath(f'./dataset/train/video_{i}.mp4') for i in range(4)]

        try:
            prepare_dataset(train_videos, dataset_path)
        except Exception as e:
            print(f"Failed to prepare dataset: {e}")
            return

        try:
            detector.train_model(dataset_path)
        except Exception as e:
            print(f"Failed to train model: {e}")
            return

    # Process test video
    try:
        print("Processing test video...")
        detector.process_video(
            './dataset/test/test_video.mp4',
            './output/result.mp4'
        )
        print("Video processing completed successfully!")
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == '__main__':
    main()