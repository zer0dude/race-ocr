"""
Convert CVAT COCO export (instances_default.json) into a simple labels.csv for OCR evaluation.

Expected:
- COCO JSON with "images" and "annotations".
- Each crop image has 1 annotation for the main bib-id text region (recommended protocol).
- The bib ID string is stored as a CVAT attribute attached to the annotation.
  CVAT COCO can represent attributes in different ways, so we handle common patterns.

Outputs:
- labels.csv with columns:
  file_name,bib_id,xtl,ytl,xbr,ybr,image_id,ann_id,category_id

Optional:
- --make_id_crops will crop the annotated ID region from the crop images into id_crops_dir/
- --rec_gt_out will write a PaddleOCR-style recognition GT file: "<relpath>\t<label>"

Usage:
  python scripts/convert_cvat_coco_to_labels_csv.py \
    --coco_json /path/to/instances_default.json \
    --out_csv /data/repos/race-ocr/data/golden/job9_val_50/labels.csv

  python scripts/convert_cvat_coco_to_labels_csv.py \
    --coco_json /path/to/instances_default.json \
    --images_dir /data/repos/race-ocr/data/golden/job9_val_50/bib_crops \
    --out_csv /data/repos/race-ocr/data/golden/job9_val_50/labels.csv \
    --make_id_crops \
    --id_crops_dir /data/repos/race-ocr/data/golden/job9_val_50/id_crops \
    --rec_gt_out /data/repos/race-ocr/data/golden/job9_val_50/rec_gt.txt \
    --pad 0.05
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


def norm_bib_id(s: str) -> str:
    s = (s or "").strip().upper()
    # remove spaces; keep A-Z0-9 only by default
    s = re.sub(r"\s+", "", s)
    return s


def clamp_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1i = max(0, min(W, int(round(x1))))
    y1i = max(0, min(H, int(round(y1))))
    x2i = max(0, min(W, int(round(x2))))
    y2i = max(0, min(H, int(round(y2))))
    if x2i < x1i:
        x1i, x2i = x2i, x1i
    if y2i < y1i:
        y1i, y2i = y2i, y1i
    return x1i, y1i, x2i, y2i


def expand_box(x1: float, y1: float, x2: float, y2: float, pad: float) -> Tuple[float, float, float, float]:
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    return x1 - px, y1 - py, x2 + px, y2 + py


def extract_bib_id_from_ann(ann: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract the bib-id string from common CVAT COCO attribute encodings.
    You may need to adapt this once you inspect your JSON structure.
    """
    # Pattern A: {"attributes": [{"name":"bib_id","value":"A123"}, ...]}
    attrs = ann.get("attributes")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").strip()
            if name.lower() in {"bib_id", "bibid", "id", "text"}:
                val = a.get("value")
                if val is not None:
                    return str(val)

    # Pattern B: {"attributes": {"bib_id": "A123", ...}}
    if isinstance(attrs, dict):
        for k in ["bib_id", "bibid", "id", "text"]:
            if k in attrs and attrs[k] is not None:
                return str(attrs[k])

    # Pattern C: {"bib_id": "..."} directly on annotation
    for k in ["bib_id", "bibid", "text", "id"]:
        if k in ann and ann[k] is not None:
            return str(ann[k])

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", type=Path, required=True, help="Path to CVAT COCO export JSON (instances_default.json)")
    ap.add_argument("--out_csv", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--images_dir", type=Path, default=None, help="Directory that contains crop images (for id-crops option)")
    ap.add_argument("--make_id_crops", action="store_true", help="Create ID-only crops from annotated bbox")
    ap.add_argument("--id_crops_dir", type=Path, default=None, help="Output dir for ID-only crops (required if --make_id_crops)")
    ap.add_argument("--rec_gt_out", type=Path, default=None, help="Optional recognition GT file (PaddleOCR style)")
    ap.add_argument("--pad", type=float, default=0.0, help="Padding fraction applied to bbox when creating id-crops")
    ap.add_argument("--fail_on_missing_bib_id", action="store_true", help="Exit nonzero if any annotation missing bib_id")

    args = ap.parse_args()

    data = json.loads(args.coco_json.read_text())

    images = data.get("images", [])
    anns = data.get("annotations", [])

    img_by_id: Dict[int, Dict[str, Any]] = {int(im["id"]): im for im in images if "id" in im}
    anns_by_img: Dict[int, list] = {}
    for ann in anns:
        img_id = int(ann.get("image_id"))
        anns_by_img.setdefault(img_id, []).append(ann)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rec_lines = []
    if args.make_id_crops:
        if args.images_dir is None:
            raise SystemExit("ERROR: --images_dir required when using --make_id_crops")
        if args.id_crops_dir is None:
            raise SystemExit("ERROR: --id_crops_dir required when using --make_id_crops")
        args.id_crops_dir.mkdir(parents=True, exist_ok=True)

    missing_bib = 0
    multi_ann = 0
    written_rows = 0

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "bib_id",
                "xtl",
                "ytl",
                "xbr",
                "ybr",
                "image_id",
                "ann_id",
                "category_id",
            ],
        )
        writer.writeheader()

        for img_id, im in img_by_id.items():
            file_name = im.get("file_name")
            if not file_name:
                continue
            img_anns = anns_by_img.get(img_id, [])
            if not img_anns:
                continue
            if len(img_anns) > 1:
                multi_ann += 1

            # Take the first annotation as "main ID region" (assuming your protocol is 1 per image)
            ann = img_anns[0]
            ann_id = ann.get("id")
            cat_id = ann.get("category_id")
            bbox = ann.get("bbox")  # [x, y, w, h] in pixels
            if not bbox or len(bbox) < 4:
                continue

            x, y, w, h = map(float, bbox[:4])
            xtl, ytl, xbr, ybr = x, y, x + w, y + h

            bib = extract_bib_id_from_ann(ann)
            bib_norm = norm_bib_id(bib or "")
            if not bib_norm:
                missing_bib += 1

            writer.writerow(
                {
                    "file_name": file_name,
                    "bib_id": bib_norm,
                    "xtl": xtl,
                    "ytl": ytl,
                    "xbr": xbr,
                    "ybr": ybr,
                    "image_id": img_id,
                    "ann_id": ann_id,
                    "category_id": cat_id,
                }
            )
            written_rows += 1

            if args.make_id_crops:
                img_path = args.images_dir / Path(file_name).name
                if not img_path.exists():
                    # Try file_name as relative path inside images_dir
                    img_path2 = args.images_dir / file_name
                    if img_path2.exists():
                        img_path = img_path2
                    else:
                        continue

                im_pil = Image.open(img_path).convert("RGB")
                W, H = im_pil.size
                ex1, ey1, ex2, ey2 = expand_box(xtl, ytl, xbr, ybr, pad=args.pad)
                x1i, y1i, x2i, y2i = clamp_box(ex1, ey1, ex2, ey2, W, H)
                if x2i <= x1i or y2i <= y1i:
                    continue

                id_crop = im_pil.crop((x1i, y1i, x2i, y2i))
                out_name = f"{Path(file_name).stem}__id.jpg"
                out_path = args.id_crops_dir / out_name
                id_crop.save(out_path, quality=95)

                if args.rec_gt_out is not None:
                    # PaddleOCR recognition label format is typically: "relative_path<TAB>label"
                    rel = out_path.relative_to(args.rec_gt_out.parent) if out_path.is_absolute() else out_path
                    rec_lines.append(f"{rel}\t{bib_norm}")

    if args.rec_gt_out is not None:
        args.rec_gt_out.parent.mkdir(parents=True, exist_ok=True)
        args.rec_gt_out.write_text("\n".join(rec_lines) + ("\n" if rec_lines else ""))

    print(
        "Done.\n"
        f"  coco_json: {args.coco_json}\n"
        f"  out_csv:   {args.out_csv}\n"
        f"  rows:      {written_rows}\n"
        f"  missing_bib_id: {missing_bib}\n"
        f"  images_with_multiple_annotations: {multi_ann}\n"
    )

    if args.fail_on_missing_bib_id and missing_bib > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
