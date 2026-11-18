"""Nối detect -> crop -> preprocess -> OCR. Không ném lỗi khi không thấy biển số."""
from . import detection, preprocessing, recognition


def run(img_bgr, roi=None, method="yolo"):
    """roi=(x,y,w,h) -> thủ công; else method "yolo" (mặc định) hoặc "contour" (dự phòng, người dùng chọn)."""
    out = {"mode": None, "det_conf": None, "box": None, "error": None, "crop": None,
           "steps": None, "ocr": {}}  # ocr: {phiên bản: (chuỗi, confidence)}
    if img_bgr is None or img_bgr.size == 0:
        out["error"] = "Ảnh rỗng hoặc không đọc được"
        return out
    if roi is None and method == "yolo" and not detection.yolo_available():
        out["mode"], out["error"] = "yolo", "Thiếu checkpoint YOLO (models/plate_yolov8_best.pt); xem docs/MODEL.md"
        return out
    if roi is not None:
        box, out["mode"] = detection.clamp_box(roi, img_bgr.shape), "manual"
    else:
        detect = detection.detect_yolo if method == "yolo" else detection.detect_contour
        box, out["mode"], out["det_conf"] = detect(img_bgr)
        box = detection.clamp_box(box, img_bgr.shape) if box else None
    if box is None:
        out["error"] = "Không phát hiện được vùng biển số"
        return out
    out["box"] = box
    out["crop"] = detection.crop(img_bgr, box)
    out["steps"] = preprocessing.preprocess(out["crop"])
    variants = {"raw": out["crop"], **out["steps"]}
    out["ocr"] = {k: recognition.read_plate(v) for k, v in variants.items()}
    return out
