from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


@dataclass
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float | None = None

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2
        y1 = self.yc - self.h / 2
        x2 = self.xc + self.w / 2
        y2 = self.yc + self.h / 2
        return x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.to_xyxy()
    bx1, by1, bx2, by2 = b.to_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def read_list(list_path: Path) -> List[Path]:
    paths = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def parse_yolo_label_file(p: Path, has_conf: bool) -> List[Box]:
    if not p.exists():
        return []
    boxes: List[Box] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # GT: cls xc yc w h
        # Pred (save_conf): cls xc yc w h conf
        if (not has_conf and len(parts) < 5) or (has_conf and len(parts) < 6):
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if has_conf else None
        boxes.append(Box(cls=cls, xc=xc, yc=yc, w=w, h=h, conf=conf))
    return boxes


def find_existing_image(dir_path: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        cand = dir_path / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Sometimes original filenames already include extension; try raw stem as full name
    cand = dir_path / stem
    return cand if cand.exists() else None


def make_side_by_side(gt_img: Path, pred_img: Path, out_path: Path, title: str | None = None):
    gt = Image.open(gt_img).convert("RGB")
    pr = Image.open(pred_img).convert("RGB")

    # Resize to same height (keep aspect)
    target_h = max(gt.height, pr.height)
    def resize_to_h(im: Image.Image, h: int) -> Image.Image:
        if im.height == h:
            return im
        w = int(im.width * (h / im.height))
        return im.resize((w, h))

    gt2 = resize_to_h(gt, target_h)
    pr2 = resize_to_h(pr, target_h)

    out = Image.new("RGB", (gt2.width + pr2.width, target_h))
    out.paste(gt2, (0, 0))
    out.paste(pr2, (gt2.width, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def greedy_match(gt: List[Box], pred: List[Box], iou_thr: float) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy matching by highest IoU (per class).
    Returns:
      matches: list of (gt_idx, pred_idx, iou)
      unmatched_gt: gt indices
      unmatched_pred: pred indices
    """
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.cls != p.cls:
                continue
            v = iou(g, p)
            candidates.append((v, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_g = set()
    used_p = set()
    matches: List[Tuple[int, int, float]] = []

    for v, gi, pi in candidates:
        if v < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, v))

    unmatched_gt = [i for i in range(len(gt)) if i not in used_g]
    unmatched_pred = [i for i in range(len(pred)) if i not in used_p]
    return matches, unmatched_gt, unmatched_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_list", required=True, help="val.txt with absolute image paths")
    ap.add_argument("--gt_dir", required=True, help="Directory with GT overlay images (from render_gt.py)")
    ap.add_argument("--pred_dirs", nargs="+", required=True, help="Prediction chunk dirs (part_00 part_01 ...)")
    ap.add_argument("--out_dir", required=True, help="Output directory for buckets + report")
    ap.add_argument("--iou_match", type=float, default=0.30, help="IoU threshold to consider a detection matched (recall proxy)")
    ap.add_argument("--iou_loc", type=float, default=0.50, help="IoU threshold for 'good localization' (below => loc error)")
    ap.add_argument("--pred_has_conf", action="store_true", help="Prediction labels include conf (save_conf)")
    ap.add_argument("--max_per_bucket", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    val_list = Path(args.val_list)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)

    # Build lookup: image stem -> (pred_img_path, pred_label_path)
    pred_img_by_stem: Dict[str, Path] = {}
    pred_lbl_by_stem: Dict[str, Path] = {}

    for d in [Path(x) for x in args.pred_dirs]:
        labels_dir = d / "labels"
        for lbl in labels_dir.glob("*.txt"):
            stem = lbl.stem
            pred_lbl_by_stem[stem] = lbl
            # Predicted rendered image should be alongside in d/ with same stem + ext
            img = find_existing_image(d, stem)
            if img:
                pred_img_by_stem[stem] = img

    # Prepare buckets
    buckets = {
        "fp": 0,
        "fn": 0,
        "loc": 0,
        "ok": 0,
        "missing_assets": 0,
    }

    report_rows = []

    max_per = args.max_per_bucket if args.max_per_bucket > 0 else None
    saved_counts = {"fp": 0, "fn": 0, "loc": 0, "ok": 0}

    for img_path in read_list(val_list):
        stem = img_path.stem

        gt_img = gt_dir / img_path.name
        pred_img = pred_img_by_stem.get(stem)

        # Derive GT label path from image path: /images/... -> /labels/... and .jpg -> .txt
        gt_lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        pred_lbl = pred_lbl_by_stem.get(stem)

        if not gt_img.exists() or pred_img is None or pred_lbl is None:
            buckets["missing_assets"] += 1
            continue

        gt_boxes = parse_yolo_label_file(gt_lbl, has_conf=False)
        pred_boxes = parse_yolo_label_file(pred_lbl, has_conf=args.pred_has_conf)

        matches, un_gt, un_pr = greedy_match(gt_boxes, pred_boxes, iou_thr=args.iou_match)

        has_fn = len(un_gt) > 0
        has_fp = len(un_pr) > 0

        # Localization error: matched but low IoU (below iou_loc)
        loc_bad = False
        if matches:
            best_iou = max(m[2] for m in matches)
            if best_iou < args.iou_loc:
                loc_bad = True
        else:
            best_iou = 0.0

        # Decide bucket (priority: fn > fp > loc > ok)
        if has_fn:
            bucket = "fn"
        elif has_fp:
            bucket = "fp"
        elif loc_bad:
            bucket = "loc"
        else:
            bucket = "ok"

        buckets[bucket] += 1

        # Save side-by-side image if within limit
        if max_per is None or saved_counts[bucket] < max_per:
            out_path = out_dir / bucket / img_path.name
            make_side_by_side(gt_img, pred_img, out_path)
            saved_counts[bucket] += 1

        report_rows.append({
            "image": str(img_path),
            "bucket": bucket,
            "n_gt": len(gt_boxes),
            "n_pred": len(pred_boxes),
            "n_matches": len(matches),
            "n_fn": len(un_gt),
            "n_fp": len(un_pr),
            "best_iou": best_iou,
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV report
    csv_path = out_dir / "error_report.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [
            "image","bucket","n_gt","n_pred","n_matches","n_fn","n_fp","best_iou"
        ])
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print("Done.")
    print("Bucket counts:", buckets)
    print("Saved side-by-side counts:", saved_counts)
    print("Report:", csv_path.resolve())


if __name__ == "__main__":
    main()
