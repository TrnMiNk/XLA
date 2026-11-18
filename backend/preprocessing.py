"""Tiền xử lý OpenCV cho crop biển số. Không sửa ảnh đầu vào."""
import cv2


def preprocess(crop_bgr, target_h=120):
    """Trả dict các ảnh trung gian: gray, enhanced (CLAHE), binary (Otsu)."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop rỗng")
    scale = target_h / crop_bgr.shape[0]
    img = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, 10)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"gray": gray, "enhanced": enhanced, "binary": binary}


def plate_face(img):
    """Cắt theo mặt biển (bỏ tem/viền dính ở mép) và ước lượng góc nghiêng. Trả (ảnh cắt, góc độ).
    Không tin cậy (vùng tìm được < 60% crop) thì trả nguyên ảnh, góc 0 — không bao giờ cắt vào ký tự vì ký tự nằm trong mặt biển."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 255).mean() < 0.5:
        bw = 255 - bw  # mặt biển là phần đa số
    k = max(3, int(0.12 * g.shape[0]) | 1)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    face = cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect), cv2.MORPH_OPEN, rect)  # lấp chỗ ký tự, bỏ tem nhỏ
    cnts, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, 0.0
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.6 * g.shape[0] * g.shape[1]:
        return img, 0.0
    (_, _), (rw, rh), ang = cv2.minAreaRect(c)
    ang = ang + 90 if rw < rh else ang
    ang = ang - 90 if ang > 45 else ang
    return img[y:y + h, x:x + w], float(ang)


def deskew(img, angle, max_deg=15.0):
    """Xoay để thẳng biển; bỏ qua góc quá nhỏ (<0,7°) hoặc quá lớn (> max_deg, không đáng tin)."""
    if abs(angle) < 0.7 or abs(angle) > max_deg:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
