import time
import numpy as np
import config

prev_positions = {}
frame_time = config.FRAME_RATE  
pixel_to_meter = config.PIXEL_TO_METER  

def estimate_speed(obj):
    obj_id = obj['id']
    box = obj['box']
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    speed = 0.0
    if obj_id in prev_positions:
        prev_center = prev_positions[obj_id]
        distance = np.linalg.norm(np.array(center) - np.array(prev_center)) * pixel_to_meter
        speed = (distance / frame_time) * 3.6  # m/s to km/h

    prev_positions[obj_id] = center
    return speed, box