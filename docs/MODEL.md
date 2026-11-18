# Checkpoint phát hiện biển số

- Nguồn: https://huggingface.co/yasirfaizahmed/license-plate-object-detection
- Revision cố định: `9d424473af93561ab1b765b99948923ec4509a89` (cập nhật 2025-02-01)
- File: `best.pt`, 6 238 179 byte, SHA-256 `d06657407970f80f1a12eb9f340661ecd003bbe44ff8feac3d5bc38845f11a94` (khớp metadata LFS của Hugging Face)
- Kiến trúc: YOLOv8 (Ultralytics 8.4.215), 1 lớp `plate`, fine-tune từ dataset `keremberke/license-plate-object-detection` (Roboflow, **không phải ảnh biển Việt Nam**).
- Giấy phép: model card ghi **Apache-2.0** cho trọng số. Thư viện Ultralytics là **AGPL-3.0** (giấy phép riêng, áp dụng cho thư viện, không phải trọng số). Dataset huấn luyện: metadata Hugging Face không nêu giấy phép; chưa xác minh.
- Rủi ro bảo mật: `.pt` là pickle, tải từ bên thứ ba có thể chứa mã độc. Đã giảm thiểu: tải đúng revision cố định, so SHA-256, quét `data.pkl` bằng `pickletools` — chỉ tham chiếu lớp torch/ultralytics/collections, không có `os`/`subprocess`/`eval`. Không đảm bảo tuyệt đối.
- Tái lập:
  ```powershell
  mkdir models
  curl.exe -L -o models/plate_yolov8_best.pt https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/9d424473af93561ab1b765b99948923ec4509a89/best.pt
  ```
  Trọng số không commit (`*.pt` trong `.gitignore`).
- Không dùng YOLO COCO mặc định; không huấn luyện lại. Chưa cần phương án thay thế vì checkpoint tải và chạy được.

## Kết quả thử thật (chưa phải đo độ chính xác)
Chỉ là quan sát thủ công trên một bộ ảnh nhỏ, bao gồm cả ảnh không được phân phối ở bản công khai; bản công khai không kèm kết quả đó. Model chưa được đo trên dữ liệu có nhãn bounding box.
