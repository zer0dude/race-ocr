from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image_list(list_path: Path) -> list[str]:
    paths: list[str] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(line)
    return paths


def validate_paths(paths: list[str]) -> list[str]:
    valid: list[str] = []
    missing = 0
    badext = 0

    for p in paths:
        fp = Path(p)
        if not fp.exists():
            missing += 1
            continue
        if fp.suffix.lower() not in IMG_EXTS:
            badext += 1
            continue
        valid.append(str(fp))

    if not valid:
        raise SystemExit("No valid images found after filtering (missing or bad extensions).")

    if missing:
        print(f"WARNING: {missing} paths missing.")
    if badext:
        print(f"WARNING: {badext} paths have unsupported extensions.")

    return valid


def main():
    ap = argparse.ArgumentParser(description="Run YOLO predictions on a list of images and save outputs.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source_list", required=True, help="Text file with one image path per line")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8, help="Inference batch size")
    ap.add_argument("--max_images", type=int, default=0, help="0 = no limit")
    ap.add_argument("--start", type=int, default=0, help="Start index into source_list (for chunking)")
    ap.add_argument("--out_dir", default="runs/predict")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_conf", action="store_true")
    ap.add_argument("--name", default="predict", help="Run name under out_dir")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    list_path = Path(args.source_list)
    if not list_path.exists():
        raise SystemExit(f"source_list not found: {list_path}")

    raw = read_image_list(list_path)

    # Apply slicing for chunking
    if args.start < 0 or args.start >= len(raw):
        raise SystemExit(f"--start {args.start} out of range (0..{len(raw)-1})")
    raw = raw[args.start :]

    if args.max_images and args.max_images > 0:
        raw = raw[: args.max_images]

    # Resolve relative paths relative to list file folder
    resolved: list[str] = []
    base = list_path.parent
    for p in raw:
        fp = Path(p)
        if not fp.is_absolute():
            fp = (base / fp).resolve()
        resolved.append(str(fp))

    sources = validate_paths(resolved)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    predict_kwargs = dict(
        source=sources,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(out_dir),
        name=args.name,
        exist_ok=True,
        verbose=False,
        stream=True,   # <-- crucial: yields results, avoids holding everything in RAM
    )
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    print(f"Predicting on {len(sources)} images (start={args.start}, max_images={args.max_images or 'all'})")
    print(f"Outputs -> {out_dir / args.name}")

    # Stream results so we don't keep all Results objects in memory
    n = 0
    with_det = 0
    for r in model.predict(**predict_kwargs):
        n += 1
        try:
            if r.boxes is not None and len(r.boxes) > 0:
                with_det += 1
        except Exception:
            pass

    print(f"Done. Images processed: {n}, images with >=1 detection: {with_det}/{n}")


if __name__ == "__main__":
    main()
