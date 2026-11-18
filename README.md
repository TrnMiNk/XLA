# PLATE VISION — Nhận diện biển số xe Việt Nam

Ứng dụng nhận diện biển số xe từ ảnh tĩnh, làm trong bài tập lớn môn **Xử lý ảnh** (PTIT).

Luồng xử lý: tải ảnh → phát hiện biển số bằng **YOLOv8** → cắt vùng biển → tiền xử lý **OpenCV** (tách mặt biển, chỉnh nghiêng, phóng ảnh; gray/CLAHE/Otsu để so sánh) → **EasyOCR** → chuỗi và độ tin cậy. Giao diện **Streamlit** gọi trực tiếp module Python `backend`, không có REST API. Phương án dự phòng: phát hiện bằng contour hoặc chọn ROI thủ công; hai chế độ này luôn được gắn nhãn và không tự động thay thế khi YOLO thất bại.

> **Giới hạn.** Chuỗi OCR hiển thị là kết quả của mô hình, **chưa được xác minh** với biển thật. EasyOCR là mô hình tổng quát, không huấn luyện riêng cho biển Việt Nam; YOLO được fine-tune trên dữ liệu không phải biển Việt Nam. Repository này không công bố độ chính xác tổng quát.

## Cấu trúc

```
backend/     detection · preprocessing · recognition · pipeline · evaluation · run_eval · run_samples · tests/
frontend/    app.py (Streamlit)
.streamlit/  config.toml
models/      nơi đặt checkpoint YOLO (không nằm trong Git)
samples/     10 ảnh Wikimedia Commons + labels.csv + SOURCES.md (tác giả, giấy phép, thay đổi)
outputs/     nơi script ghi kết quả khi chạy cục bộ (nội dung bị .gitignore)
docs/        ARCHITECTURE.md, MODEL.md (nguồn, revision, SHA-256, giấy phép checkpoint)
RELEASE_AUDIT.md   kiểm kê bản phát hành, dữ liệu không công bố, giới hạn tái lập
```

## Cài đặt và chạy (Windows PowerShell)

```powershell
git clone https://github.com/TrnMiNk/XLA.git
cd XLA
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Tải checkpoint YOLO** (đúng revision cố định) rồi kiểm SHA-256:

```powershell
curl.exe -L -o models/plate_yolov8_best.pt https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/9d424473af93561ab1b765b99948923ec4509a89/best.pt
(Get-FileHash models/plate_yolov8_best.pt -Algorithm SHA256).Hash
# phải bằng D06657407970F80F1A12EB9F340661ECD003BBE44FF8FEAC3D5BC38845F11A94
```

Nếu hash khác, **xóa tệp và không chạy**: tệp `.pt` là pickle, có thể thực thi mã khi nạp. Chi tiết nguồn, giấy phép (trọng số Apache-2.0 theo model card; thư viện Ultralytics AGPL-3.0) và rủi ro trong `docs/MODEL.md`.

**Chạy giao diện và kiểm thử** (từ thư mục gốc repository):

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

- Lần chạy EasyOCR đầu tiên tải mô hình nhận dạng nên cần mạng.
- **Thiếu checkpoint:** giao diện ẩn chế độ "YOLO tự động" và hiện cảnh báo; chế độ Contour và ROI thủ công vẫn dùng được. `pipeline.run()` ở chế độ YOLO trả thông báo lỗi rõ ràng thay vì ném ngoại lệ. Các test cần checkpoint được đánh dấu `skipped`.
- Thử nhanh: tải `samples/car_rear_vf8.jpg` lên giao diện, chọn "YOLO tự động" rồi bấm "Bắt đầu nhận diện". `samples/vf6.jpg` là ca không có biển (phải báo không phát hiện được).

Đã chạy với Python 3.13.7, numpy 2.5.5, pandas 3.1.1, easyocr 1.7.2, torch 2.15.0 (CPU), ultralytics 8.4.215, streamlit 1.67.0, pytest 9.2.0. `requirements.txt` không ghim phiên bản.

## Đánh giá (script) và giới hạn dữ liệu

`backend/run_eval.py` đo phát hiện (mức ảnh) và 4 phiên bản OCR trên `samples/labels.csv`; `backend/run_samples.py` chạy pipeline trên mọi ảnh trong `samples/`. Kết quả ghi vào `outputs/<thư mục con>/` và **không được Git theo dõi**.

Bản công khai chỉ gồm **10 ảnh có nguồn và giấy phép rõ ràng**. Bộ dữ liệu thử nghiệm gốc của bài tập lớn còn 5 ảnh khác chưa xác minh được quyền chia sẻ, nên **không được công bố**, cùng với báo cáo đầy đủ và các số liệu dựa trên bộ 15 ảnh. Vì vậy:
- Không thể tái lập số liệu trong báo cáo từ repository này.
- Chạy lại `run_eval.py` ở đây cho kết quả trên bộ 10 ảnh nhỏ, **không so sánh được** với số liệu của báo cáo; 1/6 nhãn trong đó chưa được xác nhận độc lập (`needs_user_confirmation`).

## Dữ liệu và giấy phép

- Ảnh trong `samples/`: Wikimedia Commons, CC BY-SA 3.0/4.0. Tác giả, URL, giấy phép và thay đổi (thu nhỏ về rộng 1280 px) ghi trong `samples/SOURCES.md`; dùng lại phải ghi công và giữ giấy phép ShareAlike.
- **Mã nguồn chưa được gán giấy phép**; chủ repository cần tự chọn (lưu ý Ultralytics là AGPL-3.0, tác động đến việc phân phối sản phẩm dùng nó).
