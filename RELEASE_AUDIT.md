# RELEASE_AUDIT — bản công khai chỉ gồm ứng dụng PLATE VISION

Bản sao sạch được tạo từ project gốc ngày 2026-11-18, hoàn thiện ở P6.1. Project gốc không bị thay đổi. Chưa commit, chưa push.

**Quyết định phạm vi:** bản GitHub chỉ gồm ứng dụng, mã xử lý ảnh, kiểm thử, tài liệu kỹ thuật và ảnh mẫu có giấy phép rõ ràng. Báo cáo LaTeX và dữ liệu/kết quả thực nghiệm đầy đủ được giữ riêng trong project gốc để nộp giảng viên.

## 1. Thành phần được đưa vào

| Nhóm | Đường dẫn |
|---|---|
| Ứng dụng | `backend/*.py`, `frontend/app.py`, `.streamlit/config.toml` |
| Kiểm thử | `backend/tests/test_basic.py` |
| Script đánh giá | `backend/run_eval.py`, `backend/run_samples.py` (chỉ code, không kèm kết quả) |
| Dependencies | `requirements.txt` (không ghim phiên bản) |
| Tài liệu kỹ thuật | `docs/ARCHITECTURE.md`, `docs/MODEL.md`, `README.md`, `RELEASE_AUDIT.md` |
| Ảnh mẫu | 10 ảnh Wikimedia Commons trong `samples/`, `samples/SOURCES.md`, `samples/labels.csv` (10 dòng) |
| Giữ chỗ | `models/.gitkeep`, `outputs/.gitkeep`, `.gitignore` |

## 2. Thành phần bị loại và lý do

| Loại bỏ | Lý do |
|---|---|
| `.claude/`, `CLAUDE.md`, `START_HERE.md`, `.git` cũ | Cấu hình agent, không thuộc sản phẩm |
| `.venv/`, `.pytest_cache/`, `__pycache__/` | Môi trường và cache |
| `models/plate_yolov8_best.pt` | Tải lại đúng revision và SHA-256 theo README; không commit |
| `docs/report/` (toàn bộ LaTeX, hình, `.bib`, script tạo hình/bảng, PDF) | Quyết định P6.1: báo cáo giữ riêng; PDF/hình còn nhúng ảnh `xemau*` |
| `samples/xemau.jpg` … `xemau5.jpg` | Chưa xác minh nguồn và quyền chia sẻ |
| Dòng `xemau*` trong `samples/labels.csv` (5 dòng) | Nhãn của ảnh không được phân phối; file gốc còn nguyên ở project gốc |
| Toàn bộ `outputs/` (CSV, biểu đồ, ảnh crop/hộp bao/chẩn đoán của P1.5–P4.2) | Mọi CSV tổng hợp đều có dòng `xemau*`; nhiều ảnh dẫn xuất là crop từ `xemau*`; kết quả gốc không thể tái lập từ bản công khai. Không công bố riêng một phần để tránh trộn số liệu |
| `docs/EVALUATION_P2.md`, `OCR_P41.md`, `OCR_P42.md`, `DEV_NOTES.md` | Báo cáo số liệu trên bộ 15 ảnh gồm cả `xemau*` |

## 3. Thay đổi so với project gốc (chỉ trong bản sao)

- `backend/pipeline.py`: `run()` ở chế độ YOLO trả `error` rõ ràng ("Thiếu checkpoint YOLO ...") khi thiếu checkpoint, thay vì ném `FileNotFoundError` (3 dòng, chỉ chạy khi thiếu checkpoint). Không đổi thuật toán.
- `backend/recognition.py`, `docs/MODEL.md`, `backend/tests/test_basic.py`: chỉnh chú thích/tham chiếu tới tài liệu đã bỏ.
- `backend/tests/test_basic.py`: thêm `import pytest`; thêm test `test_missing_checkpoint_reports_error`; `skipif` khi thiếu checkpoint cho `test_pipeline_fails_cleanly`; nhánh `xemau4` của test hồi quy OCR bỏ qua khi ảnh không có.
- `samples/SOURCES.md`: viết lại với tác giả, URL, giấy phép, kích thước gốc và thay đổi.
- `samples/labels.csv`: bỏ 5 dòng `xemau*`.
- `README.md`, `.gitignore`: viết lại.

## 4. Nguồn ảnh (Giai đoạn D)

Tác giả/giấy phép/kích thước gốc của **cả 10 ảnh** được đối chiếu với metadata công khai của từng trang Commons qua API (2026-11-18): 9 ảnh CC BY-SA 4.0 và 1 ảnh (`motorbike_tilted.jpg`) CC BY-SA 3.0. Tác giả: Newone (7 ảnh), Meiko, EurovisionNim, User:Vithiem. Tỉ lệ khung hình của bản trong repo khớp ảnh gốc (thu nhỏ về rộng 1280 px, riêng `hcm.jpg` giữ 658×165). **Chưa kiểm tra**: ảnh trong repo có đúng là bản thu nhỏ nguyên vẹn của tệp gốc (không cắt, không đổi hướng EXIF) hay không, chỉ dựa trên tỉ lệ khung hình và ghi chú của người thực hiện.

## 5. Kiểm thử đã thực hiện

| Kiểm tra | Kết quả |
|---|---|
| `pytest backend/tests` từ `XLA_GitHub`, **chưa có checkpoint** | 9 passed, 3 skipped |
| `pytest` trong bản sao tạm **có checkpoint** (SHA-256 `d06657…1a94` khớp `docs/MODEL.md`) | 12 passed |
| Pipeline + EasyOCR trên ảnh Wikimedia (bản sao tạm có checkpoint) | `car_rear_vf8`: YOLO conf 0,86, OCR `30K203.05`; `car_front_vf8`: conf 0,85, `30K203.05`; `car_street_front`: conf 0,78, `30K-188.75` (đối chiếu với nhãn bằng mắt) |
| Ảnh không biển `vf6.jpg` | Báo "Không phát hiện được vùng biển số" |
| Giao diện (`AppTest` + wrapper giả lập tệp tải lên), có checkpoint | `car_rear_vf8`: không ngoại lệ, các thẻ số liệu hiện (chuỗi `30K203.05`, detection 0,86); `vf6`: hiện lỗi thất bại, không ngoại lệ |
| Giao diện, **thiếu checkpoint** (từ `XLA_GitHub`) | Không ngoại lệ; ẩn chế độ YOLO, hiện cảnh báo trỏ `docs/MODEL.md` |
| `streamlit run frontend/app.py` thật từ `XLA_GitHub` | `/_stcore/health` = 200 và trang chính = 200; đã dừng server sau kiểm tra |
| `run_eval.py` trên 10 ảnh (bản sao tạm, kết quả không được đưa vào repo) | Chạy hết, ghi CSV/biểu đồ; chỉ là kiểm tra script chạy được, không phải số liệu công bố |
| Quét đường dẫn tuyệt đối, tên người dùng, email, khóa/mật khẩu trong thư mục | Không có (chỉ `.gitignore` nhắc `secrets.toml`) |
| Không có `.claude`, `CLAUDE.md`, `.pt`, `.venv`, `__pycache__` | Đúng |
| Đường dẫn trong code | Đều tương đối theo `Path(__file__)` (`models/`, `samples/`, `outputs/`) |
| `git status`, `git ls-files --others --exclude-standard` | Xem mục 7 |

## 6. Giới hạn

- **Không tái lập được báo cáo/số liệu gốc** từ repository này: bộ 15 ảnh không được công bố đủ. Chạy `run_eval.py` ở đây chỉ cho kết quả trên 10 ảnh, không so sánh được với báo cáo; 1/6 nhãn OCR trong đó chưa được xác nhận độc lập.
- Frontend chưa được thử thủ công đầy đủ trên trình duyệt trong bản sao mới; đã kiểm bằng `AppTest`, khởi động server và HTTP 200. Chưa mở trình duyệt thật.
- Lần chạy giao diện cho `car_rear_vf8` hiện OCR confidence khác (0,84) so với ảnh chụp dashboard cũ của báo cáo (0,90); chưa điều tra nguyên nhân (không ảnh hưởng chuỗi OCR).
- `requirements.txt` không ghim phiên bản; phiên bản đã chạy ghi trong README. Python 3.13.7 ngoài dải 3.10–3.12 dự định ban đầu.
- Không có giấy phép mã nguồn; chưa xác minh giấy phép dataset huấn luyện của checkpoint (`docs/MODEL.md`). Ultralytics là AGPL-3.0.
- Ảnh Wikimedia có thương hiệu VinFast và biển số của xe thật hoặc xe trưng bày; chỉ dùng cho mục đích kỹ thuật.

## 7. Git

Đã `git init` (nhánh `main`), chưa commit. Các file dự kiến commit: xem `git ls-files --others --exclude-standard` (liệt kê trong báo cáo hoàn thành).
