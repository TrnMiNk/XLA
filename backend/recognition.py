"""OCR bằng EasyOCR + chuẩn hóa chuỗi + kiểm tra định dạng. Không thay ký tự mơ hồ (A/4, B/8, 0/O...)."""
import re

import cv2

from . import preprocessing

LETTERS = "ABCDEFGHKLMNPRSTUVXYZ"  # chữ cái dùng trong biển số VN (không có I, J, O, Q, W)
ALLOW = "0123456789" + LETTERS + "-."
MIN_H = 160  # crop thấp hơn mức này được phóng (nội suy cubic) trước khi OCR; 0 = tắt (hành vi P2). Xem docs/OCR_P41.md
FACE_DESKEW = True  # tách mặt biển + chỉnh nghiêng trước OCR; False = hành vi P4.1. Xem docs/ARCHITECTURE.md
_reader = None
_CIVIL = re.compile(rf"^(\d{{2}})([{LETTERS}][{LETTERS}0-9]??)(\d{{4,5}})$")


def normalize(text):
    """Viết hoa, bỏ ký tự ngoài bảng biển số VN. Chuỗi rỗng giữ nguyên là rỗng."""
    return re.sub(r"[^0-9A-Z\-.]", "", text.upper().replace(" ", ""))


def check_format(text):
    """Kiểm tra cấu trúc biển dân dụng phổ biến: 2 số tỉnh + series (1 chữ, hoặc chữ + chữ/số) + 4-5 số.
    Trả (hợp_lệ, chuỗi_hiển_thị). Chuỗi hiển thị chỉ thêm dấu '-' và '.', KHÔNG đổi ký tự nào; không khớp -> (False, None).
    Biển quân đội, ngoại giao... không khớp cấu trúc này nên chỉ báo 'không khớp', không coi là sai."""
    s = re.sub(r"[^0-9A-Z]", "", text.upper())
    m = _CIVIL.match(s)
    if not m:
        return False, None
    prov, series, digits = m.groups()
    tail = f"{digits[:-2]}.{digits[-2:]}" if len(digits) == 5 else digits
    return True, f"{prov}{series}-{tail}"


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def order_boxes(res):
    """Nhóm các đoạn EasyOCR thành dòng theo tâm Y (trên xuống), mỗi dòng sắp theo X (trái sang phải), rồi ghép.
    res: danh sách (4 điểm, văn bản, độ tin cậy). Trả chuỗi ghép chưa chuẩn hóa."""
    items = []
    for box, text, _ in res:
        ys, xs = [p[1] for p in box], [p[0] for p in box]
        # chiều cao chữ = độ dài cạnh trái của hộp (hộp nghiêng có bao ngoài cao hơn chữ nhiều)
        text_h = ((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2) ** 0.5
        items.append(((min(ys) + max(ys)) / 2, min(xs), text_h, text))
    if not items:
        return ""
    items.sort()
    half_h = 0.5 * sorted(i[2] for i in items)[len(items) // 2]
    lines = [[items[0]]]
    for it in items[1:]:
        cy = sum(i[0] for i in lines[-1]) / len(lines[-1])
        if abs(it[0] - cy) <= half_h:
            lines[-1].append(it)
        else:
            lines.append([it])
    return "".join(i[3] for ln in lines for i in sorted(ln, key=lambda i: i[1]))


def read_plate(img):
    """Trả (chuỗi chuẩn hóa, độ tin cậy trung bình). Không đọc được -> ('', 0.0)."""
    if FACE_DESKEW:
        face, angle = preprocessing.plate_face(img)
        img = preprocessing.deskew(face, angle)
    if MIN_H and img.shape[0] < MIN_H:
        s = MIN_H / img.shape[0]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    res = get_reader().readtext(img, allowlist=ALLOW)
    if not res:
        return "", 0.0
    return normalize(order_boxes(res)), sum(r[2] for r in res) / len(res)
