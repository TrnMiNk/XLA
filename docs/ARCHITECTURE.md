# Kiến trúc

```
README.md, RELEASE_AUDIT.md
models/    checkpoint YOLO (không commit, xem docs/MODEL.md)
backend/   detection · preprocessing · recognition · pipeline · tests/
frontend/  app.py (Streamlit) -> import trực tiếp backend
samples/   ảnh mẫu + SOURCES.md      outputs/  CSV, ảnh thí nghiệm      docs/  tài liệu
```
Luồng: `pipeline.run(img_bgr, roi=None, method="yolo")` → `detection` (YOLOv8 từ `models/`, contour, hoặc ROI thủ công; không tự chuyển chế độ) → `crop` → `preprocessing` (gray, CLAHE, Otsu) → `recognition` (EasyOCR) → dict kết quả (mode, box, det_conf, crop, steps, text_raw, text_pre, error).
Không REST API, React, Node hay service riêng: Streamlit gọi hàm Python trực tiếp.
