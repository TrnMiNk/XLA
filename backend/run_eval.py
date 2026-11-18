"""P2: đánh giá YOLO (mức ảnh) và EasyOCR 4 phiên bản. Chạy: python backend/run_eval.py. Kết quả: outputs/p2/."""
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import detection, evaluation as ev, preprocessing, recognition  # noqa: E402

# python backend/run_eval.py [thư_mục_con_outputs=p2] [MIN_H] [FACE_DESKEW 0|1]: MIN_H=0 tái tạo hành vi OCR của P2; thư mục khác p2 để không ghi đè kết quả cũ
OUT = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "p2")
if len(sys.argv) > 2:
    recognition.MIN_H = int(sys.argv[2])
if len(sys.argv) > 3:
    recognition.FACE_DESKEW = sys.argv[3] == "1"  # 0 = tắt tách mặt biển/chỉnh nghiêng (hành vi P4.1)
VARIANTS = ["raw", "gray", "enhanced", "binary"]
CONF, IMGSZ = 0.25, 640  # cố định, không chỉnh theo kết quả


def outcome(has_plate, found, box_ok):
    # TP chỉ khi box đúng vị trí theo kiểm tra bằng mắt (box_visual_check); KHÔNG phải IoU (chưa có bbox ground truth)
    if has_plate and found:
        return "TP" if box_ok else "FP+FN (box sai vị trí)"
    return "FP" if found else ("FN" if has_plate else "TN")


def eval_detection(labels):
    rows, crops = [], {}
    detection.detect_yolo(cv2.imread(str(ROOT / "samples" / labels.image[0])), CONF, IMGSZ)  # khởi động model
    for r in labels.itertuples():
        img = cv2.imread(str(ROOT / "samples" / r.image))
        t = time.time()
        box, _, conf = detection.detect_yolo(img, CONF, IMGSZ)
        sec = time.time() - t
        o = outcome(r.has_valid_plate == "yes", box is not None, r.box_visual_check == "ok")
        rows.append(dict(image=r.image, category=r.category, has_valid_plate=r.has_valid_plate, found=box is not None,
                         box_xywh=box, det_conf=conf, outcome=o, sec_detect=round(sec, 3)))
        if o == "TP":
            crops[r.image] = detection.crop(img, detection.clamp_box(box, img.shape))
    return pd.DataFrame(rows), crops


def det_summary(det):
    def row(name, d):
        c = d.outcome.value_counts()
        tp, fp, fn, tn = (int(c.get(k, 0)) for k in ("TP", "FP", "FN", "TN"))
        p, r, f = ev.prf(tp, fp, fn)
        return dict(group=name, n=len(d), TP=tp, FP=fp, FN=fn, TN=tn, precision=p, recall=r, F1=f)
    return pd.DataFrame([row("ALL", det)] + [row(k, g) for k, g in det.groupby("category")])


def eval_ocr(labels, crops):
    first = next(iter(crops.values()))
    for v in [first] + [preprocessing.preprocess(first)[k] for k in VARIANTS[1:]]:
        recognition.read_plate(v)  # khởi động OCR
    rows = []
    for r in labels.itertuples():
        if r.image not in crops or not r.gt_plate:
            continue
        crop = crops[r.image]
        t = time.time()
        steps = preprocessing.preprocess(crop)
        t_pre = time.time() - t
        for v, im in {"raw": crop, **steps}.items():
            t = time.time()
            text, conf = recognition.read_plate(im)
            sec = time.time() - t
            a, b = ev.norm_orig(text), ev.norm_orig(r.gt_plate)
            c, d = ev.norm_alnum(text), ev.norm_alnum(r.gt_plate)
            rows.append(dict(image=r.image, label_status=r.label_status, variant=v, gt=r.gt_plate, ocr=text,
                             ocr_conf=round(conf, 3) if text else None, sec_ocr=round(sec, 3),
                             sec_preprocess=round(t_pre, 3) if v != "raw" else 0.0,
                             em_orig=a == b, cer_orig=round(ev.cer(a, b), 3),
                             em_alnum=c == d, cer_alnum=round(ev.cer(c, d), 3),
                             edits_alnum=ev.edit_distance(c, d), len_gt_alnum=len(d),
                             format_ok=recognition.check_format(text)[0]))
    return pd.DataFrame(rows)


def ocr_summary(o, n_gt_total):
    g = o.groupby("variant").agg(n=("image", "count"), exact_orig=("em_orig", "sum"), exact_alnum=("em_alnum", "sum"),
                                 cer_orig=("cer_orig", "mean"), cer_alnum=("cer_alnum", "mean"),
                                 mean_ocr_conf=("ocr_conf", "mean"), empty_ocr=("ocr", lambda s: int((s == "").sum())),
                                 sec_ocr=("sec_ocr", "mean"), sec_preprocess=("sec_preprocess", "mean")).reindex(VARIANTS)
    g["cer_alnum_micro"] = o.groupby("variant").edits_alnum.sum() / o.groupby("variant").len_gt_alnum.sum()  # tổng sửa / tổng ký tự GT
    g["end2end_alnum"] = g.exact_alnum / n_gt_total  # ảnh bị bỏ sót biển tính là sai
    g["EM_orig_rate"] = g.exact_orig / g.n
    g["EM_alnum_rate"] = g.exact_alnum / g.n
    return g.round(3).reset_index()


def charts(s, o, crops):
    names = ["Crop gốc", "Grayscale", "CLAHE", "Otsu"]
    x = list(range(4))
    w = 0.38
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, (a, b) in enumerate([("EM_orig_rate", "EM_alnum_rate"), ("cer_orig", "cer_alnum")]):
        ax[k].bar([i - w / 2 for i in x], s[a], w, label="giữ dấu phân cách")
        ax[k].bar([i + w / 2 for i in x], s[b], w, label="chỉ chữ/số")
    ax[0].set_title("Exact match (tỉ lệ)")
    ax[0].set_ylim(0, 1)
    ax[1].set_title("CER trung bình (thấp hơn tốt hơn)")
    ax[2].bar(x, s.mean_ocr_conf, color="gray")
    ax[2].set_title("OCR confidence TB (không phải độ chính xác)")
    ax[2].set_ylim(0, 1)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(names)
    ax[0].legend()
    fig.suptitle(f"EasyOCR trên {s.n.iloc[0]} crop do YOLO phát hiện đúng (CER = Levenshtein / len(GT))")
    fig.tight_layout()
    fig.savefig(OUT / "ocr_variants.png", dpi=130)
    plt.close(fig)

    p = o.pivot(index="image", columns="variant", values="cer_alnum")[VARIANTS]
    fig, a = plt.subplots(figsize=(7, 5))
    im = a.imshow(p.values, cmap="RdYlGn_r", vmin=0, vmax=1)
    a.set_xticks(x)
    a.set_xticklabels(["raw", "gray", "CLAHE", "Otsu"])
    a.set_yticks(range(len(p)))
    a.set_yticklabels([i[:-4] for i in p.index])
    for i in range(p.shape[0]):
        for j in x:
            a.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    a.set_title("CER (chỉ chữ/số) theo ảnh và phiên bản")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "cer_per_image.png", dpi=130)
    plt.close(fig)

    ex = [n for n in ("car_rear_vf8.jpg", "xemau3.jpg", "xemau4.jpg", "street_motorbike_far.jpg") if n in crops]
    fig, ax = plt.subplots(len(ex), 4, figsize=(13, 2.4 * len(ex)))
    for i, n in enumerate(ex):
        steps = {"raw": crops[n], **preprocessing.preprocess(crops[n])}
        for j, v in enumerate(VARIANTS):
            r = o[(o.image == n) & (o.variant == v)].iloc[0]
            ax[i, j].imshow(cv2.cvtColor(steps[v], cv2.COLOR_BGR2RGB) if v == "raw" else steps[v], cmap="gray")
            ax[i, j].axis("off")
            ax[i, j].set_title(f"{v}: [{r.ocr}] GT [{r['gt']}]" + (" OK" if r.em_alnum else ""), fontsize=8,
                               color="green" if r.em_alnum else "red")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "samples" / "labels.csv", keep_default_na=False)
    det, crops = eval_detection(labels)
    ds = det_summary(det)
    o = eval_ocr(labels, crops)
    s = ocr_summary(o, int((labels.gt_plate != "").sum()))
    det.to_csv(OUT / "detection_per_image.csv", index=False, encoding="utf-8-sig")
    ds.round(3).to_csv(OUT / "detection_summary.csv", index=False, encoding="utf-8-sig")
    o.to_csv(OUT / "ocr_per_image.csv", index=False, encoding="utf-8-sig")
    s.to_csv(OUT / "ocr_summary.csv", index=False, encoding="utf-8-sig")
    charts(s, o, crops)
    pd.set_option("display.width", 250)
    print(det[["image", "category", "found", "det_conf", "outcome"]].to_string(index=False))
    print(ds.round(3).to_string(index=False))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
