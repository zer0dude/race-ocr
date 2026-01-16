from __future__ import annotations

import argparse
import random
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())

def dump_yaml(p: Path, obj: dict) -> None:
    p.write_text(yaml.safe_dump(obj, sort_keys=False))

def list_images(img_dir: Path) -> list[Path]:
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Ultralytics YOLO dataset splits from CVAT export")
    ap.add_argument("--export_dir", type=Path, required=True,
                    help="Path to extracted CVAT Ultralytics YOLO detection export (contains data.yaml, images/train, labels/train)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Where to write split txt files + data.yaml (does not copy images)")
    ap.add_argument("--val", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", type=str, default="data.yaml", help="Output yaml filename (e.g. data_smoke.yaml)")
    args = ap.parse_args()

    export_dir = args.export_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # CVAT Ultralytics export structure (what you showed):
    # export_dir/
    #   data.yaml
    #   images/train/*.jpg
    #   labels/train/*.txt
    cvat_yaml = export_dir / "data.yaml"
    img_dir = export_dir / "images" / "train"
    lbl_dir = export_dir / "labels" / "train"

    if not cvat_yaml.exists():
        raise FileNotFoundError(f"Missing {cvat_yaml}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Missing {lbl_dir}")

    meta = load_yaml(cvat_yaml)
    names = meta.get("names")
    if names is None:
        raise ValueError("No 'names' found in export data.yaml")

    images = list_images(img_dir)
    if not images:
        raise ValueError(f"No images found in {img_dir}")

    # split
    rnd = random.Random(args.seed)
    rnd.shuffle(images)
    n = len(images)
    n_val = max(1, int(n * args.val)) if n > 1 else 0

    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    # write lists as ABSOLUTE paths (robust when txt files live elsewhere)
    def abs_line(p: Path) -> str:
        return str(p.resolve())

    (out_dir / "train.txt").write_text("\n".join(abs_line(p) for p in train_imgs) + ("\n" if train_imgs else ""))
    (out_dir / "val.txt").write_text("\n".join(abs_line(p) for p in val_imgs) + ("\n" if val_imgs else ""))

    # data yaml can omit 'path' entirely when using absolute paths, but keeping it is fine
    out_yaml = {
        "train": str((out_dir / "train.txt").resolve()),
        "val": str((out_dir / "val.txt").resolve()),
        "names": names,
    }
    dump_yaml(out_dir / args.name, out_yaml)

    print(f"Images total: {n}")
    print(f"Train: {len(train_imgs)}  Val: {len(val_imgs)}")
    print(f"Wrote: {(out_dir / args.name).resolve()}")

if __name__ == "__main__":
    main()
