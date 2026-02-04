#!/usr/bin/env python3
"""
make_bib_crops.py

Create bib crops from YOLO prediction label files, but crop from the *raw* images
(as referenced by your dataset split lists) to avoid using pre-visualized "runs" images.

Inputs:
  - src_dir: directory containing (visualized) images from YOLO predict runs (used for filenames + convenience)
  - labels_dir: directory containing YOLO prediction .txt files (default: <src_dir>/labels)
  - split list: a train/val .txt with absolute raw image paths (preferred), resolved via:
      * --split_list, OR
      * --split_dir + --subset (+ optional --use_yaml to read data.yaml)

YOLO pred label format per line:
    cls x_center y_center width height [conf]
Coordinates are normalized (0..1) relative to image width/height.

Outputs (inside out_dir):
  - bib_crops/  : cropped jpgs (from raw images)
  - meta.csv    : metadata per crop, including both vis and raw image paths

Example:
  python scripts/make_bib_crops.py \
    --src_dir /data/repos/race-ocr/runs/job9_val_pred_conf050/part_00 \
    --out_dir /data/repos/race-ocr/data/golden/job9_val_50 \
    --split_dir /data/repos/race-ocr/data/splits/job-9 \
    --subset val \
    --class_id 0 \
    --pad 0.12 \
    --min_conf 0.05
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image


def yolo_xywh_to_xyxy(
    xc: float, yc: float, w: float, h: float, W: int, H: int
) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2


def pad_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int, pad: float
) -> Tuple[int, int, int, int]:
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad, bh * pad
    x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(W, int(round(x2)))
    y2 = min(H, int(round(y2)))
    return x1, y1, x2, y2


def iter_images(src_dir: Path, exts: Iterable[str]) -> Iterable[Path]:
    for ext in exts:
        yield from src_dir.glob(f"*{ext}")


def parse_pred_lines(label_path: Path):
    for ln, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{label_path}: line {ln} malformed: {line}")
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else None
        yield cls, xc, yc, w, h, conf


def resolve_split_list(
    split_dir: Optional[Path],
    subset: str,
    split_list: Optional[Path],
    use_yaml: bool,
) -> Path:
    """
    Resolve the actual train/val list file containing absolute raw image paths.
    Priority:
      1) --split_list
      2) --split_dir + --use_yaml (reads data.yaml and takes key train/val)
      3) --split_dir/<subset>.txt
    """
    if split_list:
        return split_list

    if not split_dir:
        raise ValueError("Provide --split_list or --split_dir")

    if use_yaml:
        yml_path = split_dir / "data.yaml"
        if not yml_path.exists():
            raise FileNotFoundError(f"--use_yaml set, but data.yaml not found: {yml_path}")

        # Minimal YAML parsing without external dependency:
        # Your data.yaml is simple; we only need the 'train:' and 'val:' lines.
        train_path = None
        val_path = None
        for line in yml_path.read_text().splitlines():
            s = line.strip()
            if s.startswith("train:"):
                train_path = s.split("train:", 1)[1].strip()
            elif s.startswith("val:"):
                val_path = s.split("val:", 1)[1].strip()

        if subset == "train":
            if not train_path:
                raise ValueError(f"Could not find 'train:' in {yml_path}")
            return Path(train_path)
        else:
            if not val_path:
                raise ValueError(f"Could not find 'val:' in {yml_path}")
            return Path(val_path)

    # default convention
    return split_dir / f"{subset}.txt"


def load_raw_map(list_path: Path) -> dict:
    """
    Build a mapping from basename -> absolute raw image path,
    using the provided split list file.

    Hard-fails on duplicate basenames, because that creates ambiguity.
    """
    if not list_path.exists():
        raise FileNotFoundError(f"Split list not found: {list_path}")

    raw_by_name = {}
    duplicates = {}

    for line in list_path.read_text().splitlines():
        p = line.strip()
        if not p:
            continue
        pp = Path(p)
        name = pp.name

        if name in raw_by_name and raw_by_name[name] != pp:
            duplicates.setdefault(name, set()).update({str(raw_by_name[name]), str(pp)})
        raw_by_name[name] = pp

    if duplicates:
        ex = list(duplicates.items())[:5]
        msg = "Duplicate basenames in split list (ambiguous mapping). Examples:\n"
        for k, v in ex:
            msg += f"  {k}:\n"
            for vv in sorted(v):
                msg += f"    - {vv}\n"
        msg += "Fix by ensuring unique filenames or switching to a mapping keyed by relative path."
        raise ValueError(msg)

    return raw_by_name


def main() -> int:
    ap = argparse.ArgumentParser(description="Create bib crops from YOLO pred labels, cropping from raw images via split list.")
    ap.add_argument("--src_dir", type=Path, required=True, help="Directory containing (visualized) images from YOLO predict runs")
    ap.add_argument(
        "--labels_dir",
        type=Path,
        default=None,
        help="Directory containing YOLO label txt files (default: <src_dir>/labels)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory (creates bib_crops/ and meta.csv inside)",
    )

    # Split-based raw matching (preferred)
    ap.add_argument("--split_dir", type=Path, default=None, help="Split directory like .../data/splits/job-9")
    ap.add_argument("--subset", choices=["train", "val"], default="val", help="Which subset list to use (default: val)")
    ap.add_argument("--split_list", type=Path, default=None, help="Path to train.txt/val.txt to use directly (overrides split_dir)")
    ap.add_argument("--use_yaml", action="store_true", help="Resolve train/val list path through split_dir/data.yaml")

    # Cropping / filtering
    ap.add_argument("--class_id", type=int, default=0, help="Class ID to crop (default: 0)")
    ap.add_argument("--pad", type=float, default=0.12, help="BBox padding as fraction (default: 0.12)")
    ap.add_argument("--min_conf", type=float, default=None, help="Skip detections with conf < min_conf (default: no conf filter)")

    ap.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        help="Image extensions to include (default includes jpg/jpeg/png, case variants)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing crops (default: keep existing)")
    ap.add_argument("--max_images", type=int, default=None, help="Optional: limit number of images processed")

    args = ap.parse_args()

    src_dir: Path = args.src_dir
    labels_dir: Path = args.labels_dir if args.labels_dir else (src_dir / "labels")
    out_dir: Path = args.out_dir
    out_crops = out_dir / "bib_crops"
    out_crops.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.csv"

    if not src_dir.exists():
        print(f"ERROR: src_dir does not exist: {showp(src_dir)}", file=sys.stderr)
        return 2
    if not labels_dir.exists():
        print(f"ERROR: labels_dir does not exist: {showp(labels_dir)}", file=sys.stderr)
        return 2

    # Resolve raw mapping via split list
    try:
        list_path = resolve_split_list(args.split_dir, args.subset, args.split_list, args.use_yaml)
    except Exception as e:
        print(f"ERROR resolving split list: {e}", file=sys.stderr)
        return 2

    try:
        raw_by_name = load_raw_map(list_path)
    except Exception as e:
        print(f"ERROR loading raw map from {list_path}: {e}", file=sys.stderr)
        return 2

    images = sorted(iter_images(src_dir, args.exts))
    if args.max_images is not None:
        images = images[: args.max_images]

    fieldnames = [
        "crop_path",
        "vis_image_path",
        "raw_image_path",
        "image_stem",
        "det_idx",
        "cls",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "W",
        "H",
        "crop_W",
        "crop_H",
        "pad",
        "min_conf",
        "split_list",
    ]

    n_images_seen = 0
    n_missing_labels = 0
    n_missing_raw = 0
    n_dets_total = 0
    n_dets_kept = 0
    n_crops_written = 0

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for vis_img_path in images:
            n_images_seen += 1

            label_path = labels_dir / f"{vis_img_path.stem}.txt"
            if not label_path.exists():
                n_missing_labels += 1
                continue

            raw_img_path = raw_by_name.get(vis_img_path.name)
            if raw_img_path is None or not raw_img_path.exists():
                n_missing_raw += 1
                print(
                    f"ERROR: raw image not found for {vis_img_path.name} via split list {list_path}",
                    file=sys.stderr,
                )
                continue  # keep going, but report at end

            im = Image.open(raw_img_path).convert("RGB")
            W, H = im.size

            det_idx = 0
            for cls, xc, yc, w, h, conf in parse_pred_lines(label_path):
                if cls != args.class_id:
                    continue

                n_dets_total += 1
                if args.min_conf is not None and conf is not None and conf < args.min_conf:
                    continue

                x1, y1, x2, y2 = yolo_xywh_to_xyxy(xc, yc, w, h, W, H)
                x1, y1, x2, y2 = pad_xyxy(x1, y1, x2, y2, W, H, pad=args.pad)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = im.crop((x1, y1, x2, y2))
                crop_W, crop_H = crop.size

                conf_str = f"{conf:.3f}" if conf is not None else "na"
                crop_name = f"{vis_img_path.stem}__bib_{det_idx:03d}__c{conf_str}.jpg"
                crop_path = out_crops / crop_name

                if crop_path.exists() and not args.overwrite:
                    pass
                else:
                    crop.save(crop_path, quality=95)
                    n_crops_written += 1

                writer.writerow(
                    {
                        "crop_path": str(crop_path),
                        "vis_image_path": str(vis_img_path),
                        "raw_image_path": str(raw_img_path),
                        "image_stem": vis_img_path.stem,
                        "det_idx": det_idx,
                        "cls": cls,
                        "conf": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "W": W,
                        "H": H,
                        "crop_W": crop_W,
                        "crop_H": crop_H,
                        "pad": args.pad,
                        "min_conf": args.min_conf,
                        "split_list": str(list_path),
                    }
                )

                n_dets_kept += 1
                det_idx += 1

    print(
        "Done.\n"
        f"  split_list: {list_path}\n"
        f"  images_seen: {n_images_seen}\n"
        f"  images_missing_labels: {n_missing_labels}\n"
        f"  images_missing_raw: {n_missing_raw}\n"
        f"  dets_total_class_{args.class_id}: {n_dets_total}\n"
        f"  dets_kept_after_min_conf: {n_dets_kept}\n"
        f"  crops_written: {n_crops_written}\n"
        f"  meta_csv: {meta_path}\n"
        f"  crops_dir: {out_crops}"
    )

    # If anything is missing raw, treat as error code (so pipelines fail fast)
    return 0 if n_missing_raw == 0 else 2


def showp(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


if __name__ == "__main__":
    raise SystemExit(main())
