from __future__ import annotations

import argparse
from ultralytics import YOLO


def _str2cache(v: str):
    """
    Ultralytics accepts cache as:
      - True/'ram'
      - 'disk'
      - False
    We'll allow: ram | disk | false
    """
    v = v.strip().lower()
    if v in {"ram", "true", "1", "yes"}:
        return "ram"
    if v in {"disk"}:
        return "disk"
    if v in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("cache must be one of: ram, disk, false")


def main():
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument("--data", required=True, help="Path to Ultralytics data yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="e.g. yolo11n.pt, yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="train_run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    # Training behavior
    ap.add_argument("--cache", type=_str2cache, default="disk", help="ram | disk | false")
    ap.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    ap.add_argument("--optimizer", default="auto", help="auto, SGD, AdamW, ...")
    ap.add_argument("--lr0", type=float, default=None, help="Initial LR (ignored if optimizer=auto)")
    ap.add_argument("--lrf", type=float, default=None, help="Final LR fraction (ignored if optimizer=auto)")
    ap.add_argument("--weight_decay", type=float, default=None)

    # Augmentations (keep moderate defaults for small objects)
    ap.add_argument("--mosaic", type=float, default=0.2)
    ap.add_argument("--close_mosaic", type=int, default=10)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--copy_paste", type=float, default=0.0)
    ap.add_argument("--erasing", type=float, default=0.0)

    args = ap.parse_args()

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        cache=args.cache,
        patience=args.patience,
        optimizer=args.optimizer,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
    )

    # Only pass optional hypers if provided, otherwise let Ultralytics defaults decide
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
