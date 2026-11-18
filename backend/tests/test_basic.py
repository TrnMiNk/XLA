import numpy as np
import pytest
from backend import detection, preprocessing, recognition, pipeline


def test_clamp_box():
    assert detection.clamp_box((-5, -5, 20, 20), (100, 100, 3)) == (0, 0, 15, 15)
    assert detection.clamp_box((200, 200, 10, 10), (100, 100, 3)) is None


def test_normalize():
    assert recognition.normalize("29a-123.45 ") == "29A-123.45"
    assert recognition.normalize("???") == ""


def test_preprocess_keeps_input_and_shapes():
    img = np.full((40, 120, 3), 128, np.uint8)
    before = img.copy()
    s = preprocessing.preprocess(img)
    assert (img == before).all() and set(s) == {"gray", "enhanced", "binary"}


def test_missing_checkpoint_reports_error(monkeypatch, tmp_path):
    monkeypatch.setattr(detection, "YOLO_WEIGHTS", tmp_path / "khong_ton_tai.pt")
    r = pipeline.run(np.zeros((100, 100, 3), np.uint8))
    assert r["error"] and "checkpoint" in r["error"]


@pytest.mark.skipif(not detection.yolo_available(), reason="thiếu models/plate_yolov8_best.pt")
def test_pipeline_fails_cleanly():
    assert pipeline.run(None)["error"]
    blank = np.zeros((100, 100, 3), np.uint8)
    assert pipeline.run(blank)["error"]


import cv2
import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not detection.yolo_available(), reason="thiếu models/plate_yolov8_best.pt")
def test_yolo_finds_plate_on_whole_car_and_not_on_plateless_image():
    car = cv2.imread(str(ROOT / "samples" / "car_rear_vf8.jpg"))  # biển ở khoảng x 462-648, y 496-591
    box, mode, conf = detection.detect_yolo(car)
    x, y, w, h = box
    assert mode == "yolo" and conf > 0.25 and 440 < x + w / 2 < 670 and 480 < y + h / 2 < 610
    assert detection.detect_yolo(cv2.imread(str(ROOT / "samples" / "vf6.jpg")))[0] is None


def test_evaluation_metrics():
    from backend import evaluation as ev
    assert ev.edit_distance("kitten", "sitting") == 3
    assert ev.cer("30K203.05", "30K203.05") == 0 and ev.cer("", "ABCD") == 1.0
    assert ev.norm_orig("30K 203.05") == "30K203.05" and ev.norm_alnum("30K-203.05") == "30K20305"
    assert ev.prf(11, 1, 2)[0] == 11 / 12 and ev.prf(0, 0, 0) == (None, None, None)
    with pytest.raises(ValueError):
        ev.cer("A", "")


def test_order_boxes_two_lines_even_when_tilted_and_when_boxes_unsorted():
    from backend.recognition import order_boxes
    # tọa độ thật từ EasyOCR trên crop biển hai dòng chụp nghiêng (dòng dưới có x nhỏ hơn dòng trên)
    tilted = [([[21, 46], [160, 99], [134, 162], [-5, 109]], "203.05", 1.0), ([[51, 2], [139, 32], [116, 95], [28, 65]], "30K", 0.8)]
    assert order_boxes(tilted) == "30K203.05"
    one_line = [([[100, 0], [200, 0], [200, 30], [100, 30]], "345.67", 0.9), ([[0, 2], [95, 2], [95, 32], [0, 32]], "34A-", 0.8)]
    assert order_boxes(one_line) == "34A-345.67"
    assert order_boxes([]) == ""


def test_check_format_validates_structure_without_changing_characters():
    from backend.recognition import check_format
    assert check_format("30K203.05") == (True, "30K-203.05")
    assert check_format("34A-345.67") == (True, "34A-345.67")
    assert check_format("29-AE466.41") == (True, "29AE-466.41")
    ok, shown = check_format("8344.34567")  # đầu vào sai vẫn không bị sửa
    assert not ok and shown is None
    assert check_format("") == (False, None) and check_format("B0F33333")[0] is False
    ok, shown = check_format("37A999.99")
    assert [c for c in shown if c.isalnum()] == list("37A99999")  # chỉ thêm dấu, không đổi ký tự


def _tilted_plate(deg):
    import numpy as np
    img = np.full((130, 320, 3), 50, np.uint8)
    cv2.rectangle(img, (30, 8), (314, 122), (235, 235, 235), -1)            # mặt biển (chiếm đa số ảnh, như crop thật)
    cv2.rectangle(img, (4, 8), (30, 122), (60, 60, 230), -1)                # tem màu dính bên trái
    for x in range(60, 290, 40):
        cv2.rectangle(img, (x, 35), (x + 18, 95), (20, 20, 20), -1)         # "ký tự"
    m = cv2.getRotationMatrix2D((160, 65), deg, 1.0)
    return cv2.warpAffine(img, m, (320, 130), borderValue=(50, 50, 50))


def test_plate_face_deskew_reduces_tilt_and_keeps_characters():
    flat = _tilted_plate(0)
    face, ang = preprocessing.plate_face(flat)
    assert abs(ang) < 1 and face.shape[1] < flat.shape[1] and face.shape[1] >= 0.85 * 284  # bỏ tem trái nhưng giữ đủ mặt biển chứa ký tự
    tilted = _tilted_plate(5)
    face, ang = preprocessing.plate_face(tilted)
    assert 3 < abs(ang) < 7
    _, ang2 = preprocessing.plate_face(preprocessing.deskew(tilted, ang))
    assert abs(ang2) < abs(ang)  # đúng chiều xoay
    assert preprocessing.deskew(tilted, 40.0) is tilted  # góc quá lớn: không đáng tin, bỏ qua


def test_read_plate_returns_empty_for_no_text(monkeypatch):
    class FakeReader:
        def readtext(self, *a, **k):
            return []
    monkeypatch.setattr(recognition, "_reader", FakeReader())
    assert recognition.read_plate(_tilted_plate(0)) == ("", 0.0)


@pytest.mark.skipif(not detection.yolo_available(), reason="thiếu checkpoint")
def test_ocr_regression_on_real_crops():
    from backend import evaluation as ev
    def ocr(name):
        img = cv2.imread(str(ROOT / "samples" / name))
        box = detection.detect_yolo(img)[0]
        return recognition.read_plate(detection.crop(img, detection.clamp_box(box, img.shape)))[0]
    assert ocr("car_rear_vf8.jpg") == "30K203.05"           # ca vốn đúng không được hỏng
    if not (ROOT / "samples" / "xemau4.jpg").exists():  # ảnh không được phân phối công khai
        return
    assert ev.norm_alnum(ocr("xemau4.jpg")) == "34A34567"   # biển nghiêng + tem trái
