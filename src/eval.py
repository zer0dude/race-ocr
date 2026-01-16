from __future__ import annotations
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained weights, e.g. runs/detect/.../weights/best.pt")
    ap.add_argument("--data", required=True, help="Path to Ultralytics data yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, device=args.device)
    print(metrics)

if __name__ == "__main__":
    main()
