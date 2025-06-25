Vehicle Detection, Tracking, and Speed Estimation

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
- SORT / DeepSORT / ByteTrack
- Kalman Filter, Hungarian Algorithm

---

##  Project Structure

```bash
.
├── detector.py            # YOLOv5 vehicle detection module
├── tracker.py             # Tracking implementation
├── speed_estimator.py     # Speed calculation logic
├── detector.py            #Detection logic
├── main.py                # Main pipeline (detection, tracking, speed logic)
├── config.py              # Config structure for the project
├── __init__.py            # init file
└── README.md              # This file
