"""PLATE VISION - Streamlit UI: chỉ hiển thị, gọi thẳng backend.pipeline. Chạy từ thư mục gốc dự án."""
import hashlib
import html
import io
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import detection, pipeline, recognition  # noqa: E402

YOLO = "YOLO tự động"
AUTO = "Contour (dự phòng)"
MANUAL = "ROI thủ công"
LABELS = {"raw": "Crop gốc", "gray": "Grayscale", "enhanced": "CLAHE", "binary": "Otsu Threshold"}
NOTE_CONF = "Confidence là độ tin cậy do mô hình tự báo, không phải độ chính xác đo bằng nhãn thật; giá trị cao nhất không đồng nghĩa với kết quả đúng nhất."

st.set_page_config(page_title="PLATE VISION", layout="wide")


@st.cache_resource(show_spinner=False)
def load_reader():
    return recognition.get_reader()


@st.cache_data(show_spinner=False)
def process(data: bytes, roi, method):
    """Trả (kết quả pipeline, giây). Thời gian chỉ gồm pipeline, không gồm nạp model."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    load_reader()
    t = time.perf_counter()
    res = pipeline.run(img, roi, method)
    return res, time.perf_counter() - t


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def fit_height(img, max_h):
    """Thu nhỏ để hiển thị (không dùng cho xử lý) sao cho cao tối đa max_h."""
    h = img.shape[0]
    return img if h <= max_h else cv2.resize(img, None, fx=max_h / h, fy=max_h / h, interpolation=cv2.INTER_AREA)


def show(img, height, max_w=640):
    """Hiển thị ảnh với chiều cao mục tiêu, giữ tỉ lệ, không kéo giãn hết cột."""
    h, w = img.shape[:2]
    st.image(rgb(img), width=int(min(max_w, max(60, w * height / h))))


# ---------------- Header ----------------
st.title("PLATE VISION")
st.markdown("**Hệ thống nhận diện biển số xe Việt Nam**")
st.caption("YOLOv8 • OpenCV • EasyOCR — Bài tập lớn Xử lý ảnh, PTIT")
b1, b2, _ = st.columns([2.4, 3.6, 2.5])
if detection.yolo_available():
    b1.badge("YOLOv8: checkpoint sẵn sàng", color="green")
else:
    b1.badge("YOLOv8: thiếu checkpoint", color="red")
if st.session_state.get("ocr_loaded"):
    b2.badge("EasyOCR: đã nạp", color="green")
else:
    b2.badge("EasyOCR: chưa nạp (nạp ở lần nhận diện đầu)", color="gray")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("1. Ảnh phương tiện")
    up = st.file_uploader("Tải ảnh JPG hoặc PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.subheader("2. Chế độ nhận diện")
    modes = ([YOLO] if detection.yolo_available() else []) + [AUTO, MANUAL]
    mode = st.radio("Chế độ", modes, label_visibility="collapsed")
    if YOLO not in modes:
        st.warning("Chưa có checkpoint YOLO (models/plate_yolov8_best.pt), xem docs/MODEL.md.")
    roi_box = st.container()

if up is None:
    with st.container(border=True):
        st.markdown("#### Chưa có ảnh để nhận diện")
        st.markdown(
            "Tải ảnh JPG hoặc PNG ở thanh bên trái, chọn chế độ nhận diện, rồi bấm **Bắt đầu nhận diện**. "
            "Chế độ **YOLO tự động** phù hợp nhất với ảnh xe nguyên chiếc; chế độ **ROI thủ công** dùng khi tự động không tìm được biển số "
            "và luôn được gắn nhãn là chọn tay."
        )
    st.stop()

data = up.getvalue()
img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Không đọc được ảnh (tệp hỏng hoặc không đúng định dạng). Hãy thử một ảnh JPG hoặc PNG khác.")
    st.stop()
H, W = img.shape[:2]

roi = None
if mode == MANUAL:
    with roi_box:
        st.subheader("3. Thiết lập ROI")
        x0, x1 = st.slider("Trục X (%)", 0, 100, (0, 100))
        y0, y1 = st.slider("Trục Y (%)", 0, 100, (0, 100))
    roi = (W * x0 // 100, H * y0 // 100, W * (x1 - x0) // 100, H * (y1 - y0) // 100)

method = "yolo" if mode == YOLO else "contour"
key = (hashlib.md5(data).hexdigest(), mode, roi)
if st.sidebar.button("Bắt đầu nhận diện", type="primary", width="stretch"):
    with st.spinner("Đang nhận diện (lần đầu cần nạp model EasyOCR, có thể mất vài phút)..."):
        res_, sec_ = process(data, roi, method)
    st.session_state["ocr_loaded"] = True
    st.session_state["last"] = (key, res_, sec_)
    st.rerun()  # cập nhật trạng thái model ở header

last = st.session_state.get("last")
res, sec = (last[1], last[2]) if last and last[0] == key else (None, None)

# ---------------- Ảnh hiển thị ----------------
shown = img
if res and res["box"]:
    x, y, w, h = res["box"]
    shown = img.copy()
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), max(2, W // 300))
    if res["det_conf"] is not None:
        cv2.putText(shown, f"plate {res['det_conf']:.2f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1500), (0, 255, 0), 2)
elif mode == MANUAL:
    x, y, w, h = roi
    shown = img.copy()
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 165, 255), max(2, W // 300))

if res is None:
    st.info("Bấm **Bắt đầu nhận diện** ở thanh bên trái để xử lý ảnh với chế độ đã chọn. Chưa có kết quả nên chưa hiển thị số liệu.")
    show(shown, 420, 900)
    st.caption(f"Ảnh gốc {W}×{H}" + (" — khung cam là ROI thủ công đang chọn" if mode == MANUAL else ""))
    st.stop()

ok = res["error"] is None
fix = ""
text_raw, conf_raw = res["ocr"].get("raw", ("", 0.0)) if ok else ("", 0.0)

# ---------------- Metric cards ----------------
if ok:
    cards = st.columns(4)
    vals = [
        ("Chuỗi OCR (crop gốc, chưa xác minh)", text_raw or "Không đọc được", "Chuỗi nguyên bản EasyOCR trả về, chưa đối chiếu với biển thật."),
        ("Detection confidence", f"{res['det_conf']:.2f}" if res["det_conf"] is not None else "—",
         "Độ tin cậy của bộ phát hiện YOLO. Chế độ Contour và ROI thủ công không có giá trị này."),
        ("OCR confidence (crop gốc)", f"{conf_raw:.2f}" if text_raw else "—", "Độ tin cậy EasyOCR của chuỗi trên crop gốc."),
        ("Thời gian xử lý", f"{sec:.2f} s", "Phát hiện + tiền xử lý + 4 lần OCR trên CPU, không gồm thời gian nạp model; là một lần chạy."),
    ]
    for c, (label, val, help_) in zip(cards, vals):
        with c.container(border=True):
            st.metric(label, val, help=help_)

# ---------------- Hai cột ----------------
left, right = st.columns([1.15, 1])
with left:
    with st.container(border=True):
        st.markdown("**Ảnh phương tiện**")
        show(fit_height(shown, 440), 440, 760)
        mode_label = {"yolo": "YOLOv8 tự động", "contour": "Contour (dự phòng)", "manual": "ROI thủ công"}.get(res["mode"], mode)
        st.caption(f"Ảnh gốc {W}×{H} • Phương pháp phát hiện: {mode_label}")
        with st.expander("Xem ảnh kích thước lớn"):
            st.image(rgb(shown), width="stretch")

with right:
    with st.container(border=True):
        st.markdown("**Kết quả nhận diện**")
        if res["mode"] == "manual":
            st.warning("Chế độ ROI thủ công: vùng biển số do người dùng chọn, không phải phát hiện tự động.")
        if not ok:
            st.error(f"Thất bại: {res['error']}.")
            if mode != MANUAL:
                st.info("Gợi ý: thử chế độ khác ở thanh bên, hoặc chọn **ROI thủ công** và kéo hai thanh trượt quanh biển số.")
        else:
            show(res["crop"], 130, 420)
            st.caption("Crop biển số (hiển thị phóng đại nếu nhỏ)")
            det_txt = f"{res['det_conf']:.2f}" if res["det_conf"] is not None else "không có (chế độ này không có detection confidence)"
            st.caption(f"Detection confidence: {det_txt} • OCR confidence (crop gốc): {conf_raw:.2f}" if text_raw else f"Detection confidence: {det_txt}")
            if text_raw:
                st.caption("Chuỗi OCR nguyên bản (crop gốc)")
                st.html(f'<div style="font-size:2.1rem;font-weight:700;letter-spacing:0.06em;margin:0 0 0.4rem 0">{html.escape(text_raw)}</div>')
                valid, shown_fmt = recognition.check_format(text_raw)
                if valid:
                    st.caption(f"Chuỗi chuẩn hóa (chỉ thêm dấu phân cách, không đổi ký tự): **{shown_fmt}**")
                    st.info("OCR có trả chuỗi và chuỗi khớp cấu trúc biển dân dụng phổ biến. Đây chỉ là kiểm tra cấu trúc, **chưa xác minh chuỗi đúng hay sai** so với biển thật.")
                else:
                    st.warning("Chuỗi OCR **không khớp** cấu trúc biển dân dụng phổ biến (2 số tỉnh + chữ series + 4–5 số), nên có thể OCR đã đọc sai ký tự "
                               "(hoặc đây là biển loại khác). Hãy đối chiếu thủ công với ảnh; chuỗi không được tự sửa.")
            else:
                st.warning("Đã tìm thấy vùng biển số nhưng OCR không đọc được ký tự nào trên crop gốc. Xem các phiên bản tiền xử lý ở tab So sánh OCR.")
            st.caption("Chuỗi hiển thị là kết quả trên crop gốc; các phiên bản khác có thể khác, xem tab So sánh OCR.")
            fix = st.text_input("Sửa thủ công chuỗi biển số (tùy chọn)", key=f"fix_{key[0]}_{mode}",
                                placeholder="Nhập chuỗi đúng nếu OCR sai; không được tự xác minh", help="Chuỗi bạn nhập chỉ được ghi vào cột riêng của CSV, không thay kết quả OCR.")

if not ok:
    st.stop()

# ---------------- Phân tích ảnh ----------------
imgs = {"raw": res["crop"], **res["steps"]}
rows = [{"Phiên bản": LABELS[k], "Chuỗi OCR": t or "(rỗng)", "OCR confidence": round(c, 3) if t else None,
         "Khớp cấu trúc biển": ("Có" if recognition.check_format(t)[0] else "Không") if t else "—"}
        for k, (t, c) in res["ocr"].items()]
df = pd.DataFrame(rows)

st.markdown("#### Phân tích các bước xử lý ảnh")
tabs = st.tabs([LABELS[k] for k in imgs] + ["So sánh OCR"])
for tab, (k, v) in zip(tabs, imgs.items()):
    with tab:
        c_img, c_txt = st.columns([2, 1])
        with c_img:
            show(v, 200, 640)
        with c_txt:
            t, c = res["ocr"][k]
            st.metric("Chuỗi OCR", t or "(rỗng)")
            st.metric("OCR confidence", f"{c:.2f}" if t else "—")
with tabs[-1]:
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"OCR confidence": st.column_config.NumberColumn(format="%.3f")})
    if all(not t for t, _ in res["ocr"].values()):
        st.warning("OCR không đọc được ký tự nào trên mọi phiên bản.")
    st.caption(NOTE_CONF)

# ---------------- Xuất dữ liệu ----------------
st.markdown("#### Xuất kết quả")
full = df.assign(**{"Detection confidence": res["det_conf"] and round(res["det_conf"], 3), "Chế độ": res["mode"], "Chuỗi người dùng sửa": fix.strip()})
stem = Path(up.name).stem
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    for k, v in {"box": shown, **imgs}.items():
        z.writestr(f"{stem}_{k}.png", png(v))
    z.writestr(f"{stem}_result.csv", full.to_csv(index=False))
c1, c2, c3 = st.columns(3)
c1.download_button("Tải CSV", full.to_csv(index=False).encode("utf-8-sig"), f"{stem}_result.csv", "text/csv", width="stretch")
c2.download_button("Tải ZIP (ảnh + CSV)", buf.getvalue(), f"{stem}_result.zip", "application/zip", width="stretch")
if c3.button("Lưu vào outputs/", width="stretch"):
    out = ROOT / "outputs" / stem
    out.mkdir(parents=True, exist_ok=True)
    for k, v in {"box": shown, **imgs}.items():
        (out / f"{k}.png").write_bytes(png(v))
    full.to_csv(out / "result.csv", index=False, encoding="utf-8-sig")
    st.success(f"Đã lưu vào {out}")
