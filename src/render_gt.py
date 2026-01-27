from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_list", required=True, help="val.txt with image paths")
    ap.add_argument("--out_dir", required=True, help="where to save GT overlay images")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = Path(args.source_list)
    paths = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]

    for p in paths:
        img_path = Path(p)
        if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        # map images/... -> labels/... and .jpg -> .txt
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = parts[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)

        out_path = out_dir / img_path.name
        im.save(out_path)

    print(f"Saved GT overlays to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
