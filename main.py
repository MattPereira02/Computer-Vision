import cv2
from detector import load_model, detect_vehicles
from tracker import initialize_tracker, update_tracker
from speed_estimator import estimate_speed
import config

# Load YOLO model
model = load_model()

# Initialize SORT tracker
tracker = initialize_tracker()

# Video input
cap = cv2.VideoCapture("videos/highway.mp4")

# Speed limit (e.g., 80 km/h)
SPEED_LIMIT = config.SPEED_LIMIT_KMH

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('output/output.mp4', fourcc, fps, frame_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    detections = detect_vehicles(model, frame)

    # Update tracker
    tracked_objects = update_tracker(tracker, detections)

    # Estimate speed and annotate
    for obj in tracked_objects:
        speed, box = estimate_speed(obj)
        x1, y1, x2, y2 = box

        color = (0, 255, 0)
        label = f"{speed:.1f} km/h"
        if speed > SPEED_LIMIT:
            color = (0, 0, 255)
            label += " OVERSPEED"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow("Speed Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
