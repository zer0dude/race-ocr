"""
predict.py — Run Ultralytics YOLO inference on an explicit list of images.

This script is designed for *batch prediction* over a known set of image paths
(e.g., a validation split exported from CVAT or a custom dataset manifest),
while keeping memory usage stable by streaming results from Ultralytics.

Key features
------------
- Reads a plain-text file (--source_list) containing one image path per line.
  - Blank lines and comments starting with '#' are ignored.
  - Paths may be absolute or relative. Relative paths are resolved relative to
    the directory containing the list file.
- Validates paths:
  - Filters out missing files and unsupported extensions.
  - Exits if nothing valid remains after filtering.
- Chunking support:
  - --start and --max_images allow processing slices of the list, enabling
    parallelization or repeated runs without editing the list file.
- Output management:
  - Writes standard Ultralytics prediction artifacts under:
      <out_dir>/<name>/
    including images with drawn boxes (save=True) and optionally label txt files.
- Memory safety:
  - Uses stream=True to iterate over predictions without keeping all `Results`
    objects in RAM. This is important for large lists.

Expected outputs (Ultralytics defaults)
---------------------------------------
Inside <out_dir>/<name>/ you can typically expect:
- annotated images (if save=True)
- labels/ (if --save_txt is set)
- other Ultralytics metadata depending on version

Notes
-----
- This script uses Ultralytics' `YOLO(...).predict()` with a list of sources.
  The behavior of output files and directory structure can vary slightly
  between Ultralytics versions.
- `--device` accepts values understood by Ultralytics (e.g., "0", "cpu").
- `--imgsz` is optional; if omitted, Ultralytics will use a model/version default.

Example `source_list` file
--------------------------
# data/splits/job10_val.txt
/data/images/img_0001.jpg
/data/images/img_0002.jpg
relative/path/img_0003.png

# Blank lines and comments are ignored.

Example usage
-------------
Run predictions on an entire list:
    python -m src.predict \\
      --weights runs/job10_y11s/weights/best.pt \\
      --source_list data/splits/job10_val.txt \\
      --conf 0.25 --iou 0.7 \\
      --device 0 \\
      --out_dir runs/predict \\
      --name job10_val

High-recall sweep (low confidence), saving YOLO txt labels + confidences:
    python -m src.predict \\
      --weights runs/job10_y11s/weights/best.pt \\
      --source_list data/splits/job10_val.txt \\
      --conf 0.05 \\
      --save_txt --save_conf \\
      --out_dir runs/predict \\
      --name job10_val_conf005

Chunking (process images 2000..2999):
    python -m src.predict \\
      --weights runs/job10_y11s/weights/best.pt \\
      --source_list data/splits/job10_val.txt \\
      --start 2000 \\
      --max_images 1000 \\
      --out_dir runs/predict \\
      --name job10_val_chunk_2

Batch size tuning (GPU memory dependent):
    python -m src.predict \\
      --weights runs/job10_y11s/weights/best.pt \\
      --source_list data/splits/job10_val.txt \\
      --batch 16 \\
      --device 0 \\
      --out_dir runs/predict \\
      --name job10_val_b16

"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Supported image file extensions for filtering input lists.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image_list(list_path: Path) -> list[str]:
    """
    Read an image list file and return the raw path strings.

    The list file is expected to contain one path per line.
    - Lines that are empty or start with '#' are ignored.
    - Paths may be absolute or relative (relative paths are resolved later).

    Parameters
    ----------
    list_path:
        Path to a text file containing one image path per line.

    Returns
    -------
    list[str]
        Raw path strings (not yet resolved to absolute paths).
    """
    paths: list[str] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(line)
    return paths


def validate_paths(paths: list[str]) -> list[str]:
    """
    Filter and validate candidate image paths.

    Validation rules:
    - File must exist.
    - File extension must be in IMG_EXTS.

    Warnings are printed for:
    - missing paths
    - unsupported extensions

    If no valid images remain, the script exits with a SystemExit.

    Parameters
    ----------
    paths:
        Candidate image paths (typically absolute by the time this is called).

    Returns
    -------
    list[str]
        Validated file paths as strings (ready to pass to Ultralytics as `source`).
    """
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
    """
    CLI entrypoint for running YOLO predictions over an explicit list of images.

    This function:
    1) Parses CLI args
    2) Loads and slices the source list (--start, --max_images)
    3) Resolves relative paths relative to the list file directory
    4) Validates paths (existence + extension)
    5) Runs Ultralytics YOLO `.predict()` with `stream=True` to keep memory stable
    6) Prints a small summary: processed count + images with >=1 detection

    Example
    -------
    Basic run:
        python -m src.predict \\
          --weights runs/job10_y11s/weights/best.pt \\
          --source_list data/splits/job10_val.txt \\
          --out_dir runs/predict \\
          --name job10_val

    High-recall (low conf), plus YOLO txt outputs:
        python -m src.predict \\
          --weights runs/job10_y11s/weights/best.pt \\
          --source_list data/splits/job10_val.txt \\
          --conf 0.05 \\
          --save_txt --save_conf \\
          --out_dir runs/predict \\
          --name job10_val_conf005

    Chunking:
        python -m src.predict \\
          --weights runs/job10_y11s/weights/best.pt \\
          --source_list data/splits/job10_val.txt \\
          --start 5000 \\
          --max_images 1000 \\
          --out_dir runs/predict \\
          --name job10_val_chunk_5

    Notes
    -----
    - `--start` is an index into the list after comment/blank filtering.
      The script exits if --start is out of range.
    - `--max_images=0` means "no limit".
    - Outputs are written to: <out_dir>/<name>/ (Ultralytics `project` + `name`).
    - `stream=True` is crucial for large runs; it yields results instead of
      storing all `Results` objects in memory.
    """
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
