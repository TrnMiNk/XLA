"""Chạy pipeline trên mọi ảnh trong samples/, lưu ảnh có box, crop, CSV (cần ground truth mới đo được độ chính xác)."""
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import pipeline  # noqa: E402


def main(out=ROOT / "outputs" / "p16", method="yolo"):
    out.mkdir(parents=True, exist_ok=True)
    pipeline.run(cv2.imread(str(ROOT / "samples" / "hcm.jpg")), None, method)  # khởi động model, không tính thời gian
    rows = []
    for f in sorted((ROOT / "samples").glob("*.jpg")):
        img = cv2.imread(str(f))
        t = time.time()
        r = pipeline.run(img, None, method)
        sec = time.time() - t
        row = {"image": f.name, "method": method, "found": r["box"] is not None, "box_xywh": r["box"],
               "det_conf": r["det_conf"], "error": r["error"], "sec_total": round(sec, 2)}
        if r["box"]:
            x, y, w, h = r["box"]
            vis = img.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), max(2, img.shape[1] // 250))
            cv2.imwrite(str(out / f"{f.stem}_{method}_box.jpg"), vis)
            cv2.imwrite(str(out / f"{f.stem}_{method}_crop.png"), r["crop"])
            for k, (t_, c_) in r["ocr"].items():
                row[f"ocr_{k}"], row[f"ocr_conf_{k}"] = t_, round(c_, 3)
        rows.append(row)
        print(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith(("image", "method", "found")), k))
    with open(out / f"results_{method}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(method=sys.argv[1] if len(sys.argv) > 1 else "yolo")
