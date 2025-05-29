import torch
import cv2
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

model = None
imgsz = 640
device = None

def load_model(weights='yolov5s.pt'):
    global model, device
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    return model

def detect_vehicles(model, frame):
    img = cv2.resize(frame, (imgsz, imgsz))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[2, 3, 5, 7])[0]  # cars, trucks, buses

    detections = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape[:2]).round()
        for *box, conf, cls in pred:
            x1, y1, x2, y2 = map(int, box)
            detections.append([x1, y1, x2, y2, float(conf)])
    return detections
