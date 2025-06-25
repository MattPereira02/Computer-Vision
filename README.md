# Computer Vision Vehicle Detection, Tracking, and Speed Estimation

This project implements a real-time system to detect vehicles in highway video footage, track them across frames, and estimate their speeds in km/h. The system visually highlights vehicles exceeding a specified speed limit. It uses **YOLOv5** for object detection, **SORT** for tracking (with optional support for DeepSORT and ByteTrack), and **OpenCV** for video processing.

---

##  Features

-  **Vehicle Detection**: Real-time vehicle detection using YOLOv5.
-  **Object Tracking**: Implements SORT, with optional modules for DeepSORT and ByteTrack.
-  **Speed Estimation**: Calculates speed using pixel displacement and frame rate.
-  **Speeding Alert**: Highlights speeding vehicles with red bounding boxes.
-  **Modular Codebase**: Easy to extend or swap detection/tracking modules.
-  **Lightweight**: Optimized to run in real time on mid-range hardware.

---

##  Technologies Used

- Python 3.x
- OpenCV
- YOLOv5 (Ultralytics)
- NumPy
- SORT
- Kalman Filter, Hungarian Algorithm

---

##  Project Structure

```bash
.
├── detector.py              # YOLOv5 vehicle detection module
├── tracker.py               # Tracking implementation
├── speed_estimator.py       # Speed calculation logic
├── detector.py              # Detection logic
├── main.py                  # Main pipeline (detection, tracking, speed logic)
├── config.py                # Config structure for the project
├── __init__.py              # init file
├── videos                   # Folder where input video is placed
├── output                   # Folder where output video is generated
├── yolov5s.pt               # Yolov5
├──README.md                 # This file
├── sort                     # Folder with sort related files
      ├── __init__.py
      ├── kalman_filter.py   # Kalman filter
      └──sort.py             # Sort File
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/MattPereira02/Computer-Vision.git
cd Computer Vision
```
2. Install requirements found in requirements.txt

## Usage

1. Place video in the videos folder
2. Run projet
```bash
python main.py
```
## Sample output frame
![image](https://github.com/user-attachments/assets/f3399121-8c2f-4926-818b-9816ff900b73)
