"""Tìm vùng biển số. Chưa có checkpoint chuyên biển số -> dùng contour (dự phòng) hoặc ROI thủ công."""
from pathlib import Path

import cv2
import numpy as np

YOLO_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "plate_yolov8_best.pt"
_yolo = None


def clamp_box(box, shape):
    """Đưa (x, y, w, h) về trong ảnh; trả None nếu rỗng."""
    x, y, w, h = (int(v) for v in box)
    H, W = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w].copy()


def detect_contour(img):
    """Contour hình chữ nhật có tỉ lệ giống biển số. Best-effort; trả (box, "contour", None) hoặc (None, "contour", None)."""
    H, W = img.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 75, 75)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / h
        # biển 1 dòng ~4.3:1, 2 dòng ~1.3:1; diện tích 0.5%–40% ảnh
        if not (0.005 * W * H < area < 0.4 * W * H) or not (1.0 < ar < 5.5):
            continue
        fill = cv2.contourArea(c) / area  # độ "chữ nhật" của contour
        score = fill * area
        if fill > 0.3 and score > best_score:
            best, best_score = (x, y, w, h), score
    return best, "contour", None


def yolo_available():
    return YOLO_WEIGHTS.exists()


def detect_yolo(img, conf=0.25, imgsz=640):
    """YOLOv8 chuyên biển số (models/, nguồn: docs/MODEL.md). Trả (box, "yolo", detection_conf) hoặc (None, "yolo", None)."""
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_WEIGHTS))  # tải 1 lần, dùng lại
    boxes = _yolo.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0].boxes
    if len(boxes) == 0:
        return None, "yolo", None
    i = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), "yolo", float(boxes.conf[i])
