from sort.sort import Sort
import numpy as np

tracker = None
def initialize_tracker():
    global tracker
    tracker = Sort()
    return tracker

def update_tracker(tracker, detections):
    valid_detections = []
    for detection in detections:
        x1, y1, x2, y2, score = detection
        if x2 > x1 and y2 > y1:  # Check if bounding box is valid
            valid_detections.append(detection)
    tracked = tracker.update(np.array(valid_detections))
    results = []
    for d in tracked:
        x1, y1, x2, y2, obj_id = map(int, d[:5])
        results.append(({"id": obj_id, "box": [x1, y1, x2, y2]}))
    return results