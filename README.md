# Pedestrian Tracking System leveraging optimum traffic light

A computer vision system that tracks pedestrians in video footage and measures system performance metrics.

Demo video
[https://youtu.be/-b4mw582Sy0 ](https://youtu.be/-b4mw582Sy0) <br />
[https://youtu.be/E1zu8EKmnNI](https://youtu.be/E1zu8EKmnNI)

## Features

- Pedestrian detection using YOLOv8
- Multi-object tracking with DeepSORT
- Zone-based pedestrian monitoring
- Movement direction and speed calculation
- Estimating the crossing time based on the given trigger
- Based on the estimation trigger fraffic light signal
- Comprehensive performance benchmarking

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- DeepSORT
- Other dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/vithanagevms/dynamic-predestrian-crossing-timing.git
cd dynamic-predestrian-crossing-timing

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 weights (optional - will auto-download if not present)
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

```bash
# Basic usage
python pedestrian_tracker_v2.py --config video_settings.json

# With benchmarking enabled
python pedestrian_tracker_v2.py --benchmark --benchmark-output results/benchmark_data.csv

# Additional options
python pedestrian_tracker_v2.py --model yolov8s.pt --skip 3 --benchmark
```

## Configuration

Initial run the system will open up the window to setup the video input (recorded or live) and crossing markups. 

Create a `video_settings.json` file with the following structure:

```json
[
  {
    "path": "path/to/your/video.mp4",
    "top_left": [100, 100],
    "top_right": [540, 100],
    "bottom_right": [540, 260],
    "bottom_left": [100, 260],
    "rotate": 0,
    "distance": "1000",
    "videosize": [640, 360]
  }
]
```

## Traffic Light System
The system implements a dynamic traffic light control mechanism that responds to pedestrian presence and movement:
bounding boxes of pedestrians were colored in greeen. Red if they are in zone, orange when estimating the slowest. 

- BLINKING_GREEN state: Initial state when pedestrians are first detected entering the crossing zone
- COUNTDOWN state: After monitoring for 10 seconds, calculates completion time for the slowest pedestrian
- GREEN state: When all pedestrians have cleared the crossing zone

The traffic light visualization appears in the top-left corner of the display, showing the current state and countdown timer when applicable. The system automatically adjusts timing based on real-time pedestrian tracking data.


## Benchmarking

The system measures the following performance metrics:

- Detection Latency: Time for YOLO to detect pedestrians
- Tracking Update Latency: Time for DeepSORT to track objects
- Prediction Computation: Time to calculate pedestrian metrics
- Signal Control Decision: Time for system state updates
- Total System Response: End-to-end latency
- Frame Processing Time: Overall per-frame processing time

## Keyboard Controls

- `q`: Quit the application
- `p`: Pause/resume the video
- `r`: Reset the traffic light system
- `s`: Save a snapshot of the current frame
- `b`: Export benchmark data (when benchmarking is enabled)

## License

[MIT License](LICENSE)