"""
Run PaddleOCR on bib crops and evaluate against labels.csv generated from CVAT COCO.

Outputs:
- per-image predictions CSV
- summary JSON (accuracy etc.)

Heuristic to pick "best" candidate text:
- normalize text to A-Z0-9 (strip spaces)
- prefer higher OCR confidence
- prefer larger text area
- prefer digit-heavy strings (IDs often mostly digits)
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# PaddleOCR imports
from paddleocr import PaddleOCR


ALNUM_RE = re.compile(r"^[A-Z0-9]+$")


def norm_text(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    # keep only A-Z0-9
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def box_area(quad: List[List[float]]) -> float:
    # quad = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return max(xs) - min(xs) * (max(ys) - min(ys))


def quad_bbox(quad: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    d = sum(c.isdigit() for c in s)
    return d / len(s)


def pick_best_candidate(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    cands: list of {text_norm, text_raw, conf, quad, area, digit_ratio}
    """
    if not cands:
        return {"text_norm": "", "text_raw": "", "conf": 0.0}

    # Filter to plausible strings first
    plausible = [c for c in cands if c["text_norm"] and ALNUM_RE.match(c["text_norm"])]
    if not plausible:
        plausible = cands

    def score(c):
        # tuned for bib IDs: confidence primary, then area, then digit-heavy
        return (
            float(c.get("conf", 0.0)) * 10.0
            + float(c.get("area", 0.0)) * 0.0005
            + float(c.get("digit_ratio", 0.0)) * 0.5
            - abs(len(c.get("text_norm", "")) - 4) * 0.05  # mild preference for ~4 length IDs
        )

    return max(plausible, key=score)


def load_labels(labels_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map file_name -> {bib_id, bbox}
    """
    out = {}
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = Path(r["file_name"]).name
            out[fn] = {
                "bib_id": r.get("bib_id", ""),
                "bbox": (
                    float(r.get("xtl", 0)),
                    float(r.get("ytl", 0)),
                    float(r.get("xbr", 0)),
                    float(r.get("ybr", 0)),
                ),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=Path, required=True, help="Directory with bib crop images")
    ap.add_argument("--labels_csv", type=Path, required=True, help="labels.csv produced by convert script")
    ap.add_argument("--out_csv", type=Path, required=True, help="Per-image prediction output CSV")
    ap.add_argument("--summary_json", type=Path, required=True, help="Summary JSON output")

    ap.add_argument("--use_gpu", action="store_true", help="Use GPU (Paddle must be installed with GPU support)")
    ap.add_argument("--use_angle_cls", action="store_true", help="Enable angle classifier (helps rotated text)")
    ap.add_argument("--lang", default="en", help="PaddleOCR language (default: en)")
    ap.add_argument("--max_images", type=int, default=None, help="Limit number of images")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"])

    args = ap.parse_args()

    labels = load_labels(args.labels_csv)

    imgs = []
    for ext in args.exts:
        imgs.extend(args.images_dir.glob(f"*{ext}"))
    imgs = sorted(imgs)
    if args.max_images is not None:
        imgs = imgs[: args.max_images]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_angle_cls=args.use_angle_cls,
        lang=args.lang,
        use_gpu=args.use_gpu,
    )

    rows = []
    n = 0
    n_match = 0
    n_missing_gt = 0
    n_empty_pred = 0

    for img_path in imgs:
        fn = img_path.name
        gt = labels.get(fn)
        if gt is None:
            n_missing_gt += 1
            continue

        gt_id = norm_text(gt["bib_id"])
        result = ocr.ocr(str(img_path), cls=args.use_angle_cls)

        # PaddleOCR returns a list per image; common structure:
        # result = [ [ [quad, (text, conf)], [quad, (text, conf)], ... ] ]
        dets = result[0] if result else []
        cands = []
        for det in dets:
            try:
                quad = det[0]
                text_raw, conf = det[1][0], float(det[1][1])
            except Exception:
                continue
            tnorm = norm_text(text_raw)
            x1, y1, x2, y2 = quad_bbox(quad)
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            cands.append(
                {
                    "text_raw": text_raw,
                    "text_norm": tnorm,
                    "conf": conf,
                    "quad": quad,
                    "area": area,
                    "digit_ratio": digit_ratio(tnorm),
                }
            )

        best = pick_best_candidate(cands)
        pred_id = best.get("text_norm", "")
        pred_conf = float(best.get("conf", 0.0))

        is_match = (pred_id == gt_id) and (gt_id != "")
        n += 1
        n_match += int(is_match)
        if not pred_id:
            n_empty_pred += 1

        # keep top-5 candidates by confidence for debugging
        top5 = sorted(cands, key=lambda c: float(c.get("conf", 0.0)), reverse=True)[:5]

        rows.append(
            {
                "file_name": fn,
                "gt_bib_id": gt_id,
                "pred_bib_id": pred_id,
                "match": int(is_match),
                "pred_conf": pred_conf,
                "top5_candidates_json": json.dumps(
                    [{"t": c["text_norm"], "raw": c["text_raw"], "conf": c["conf"]} for c in top5],
                    ensure_ascii=False,
                ),
            }
        )

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "gt_bib_id", "pred_bib_id", "match", "pred_conf", "top5_candidates_json"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images_evaluated": n,
        "exact_match": n_match,
        "exact_match_acc": (n_match / n) if n else 0.0,
        "missing_gt": n_missing_gt,
        "empty_pred": n_empty_pred,
        "use_gpu": bool(args.use_gpu),
        "use_angle_cls": bool(args.use_angle_cls),
        "lang": args.lang,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2))

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
