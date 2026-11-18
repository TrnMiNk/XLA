"""Metric từ nhãn thật: CER, exact match, hai cách chuẩn hóa. Không sửa nội dung OCR."""
import re


def norm_orig(s):
    """Giữ dấu phân cách (-, .); chỉ bỏ khoảng trắng (biển 2 dòng) và viết hoa."""
    return re.sub(r"\s+", "", str(s or "")).upper()


def norm_alnum(s):
    """Chỉ giữ chữ/số viết hoa."""
    return re.sub(r"[^0-9A-Z]", "", str(s or "").upper())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    """Edit distance / len(gt). gt rỗng không có nghĩa -> ValueError. OCR rỗng => CER = 1.0 (hợp lệ)."""
    if not gt:
        raise ValueError("cần ground truth không rỗng")
    return edit_distance(pred, gt) / len(gt)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return p, r, f
