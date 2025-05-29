VIDEO_PATH = "videos/highway.mp4"  # Path to input video
YOLO_MODEL_PATH = "yolo/yolov8n.pt"  # YOLOv8 model file
CONFIDENCE_THRESHOLD = 0.5  # Object detection confidence
NMS_THRESHOLD = 0.4  # Non-max suppression threshold
SPEED_LIMIT_KMH = 60  # Speed limit in km/h
FRAME_RATE = 1 / 30 # FPS of the video
PIXEL_TO_METER = 0.05 # change depending on your video scale