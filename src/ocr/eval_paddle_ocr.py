#!/usr/bin/env python3
"""
Evaluate PaddleOCR (PP-OCRv5 via paddleocr.PaddleOCR) on a crop dataset.

This script is tailored for the "race-ocr" style setup where:
- labels.csv contains the ground-truth text per crop (one correct solution per crop).
- meta.csv contains absolute crop paths (and other metadata).
- OCR may output 0, 1, or many detection boxes per crop.
- We select the OCR result of the *largest* detected box as the single prediction.

Outputs are written under:
    <repo_root_or_cwd>/runs/ocr/<run_name>/

Artifacts:
- pred_json/<stem>.json         Full OCR output + chosen box/text (+ GT + errors if any)
- pred_imgs/<stem>.jpg          Visualization (all boxes + chosen highlighted) (best-effort)
- predictions.csv               labels.csv extended with predicted values + selected bbox + errors
- prediction_summary.json       Basic evaluation metrics

Dependencies:
- paddleocr
- pandas
- pillow
- numpy (recommended)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from paddleocr import PaddleOCR


# ----------------------------- Data structures ----------------------------- #

@dataclass
class SelectedPrediction:
    """Represents the single chosen OCR prediction for a crop."""
    idx: Optional[int]
    text: str
    score: Optional[float]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    poly: Optional[List[List[int]]]  # list of [x,y]


# ----------------------------- Utility functions ----------------------------- #

def now_run_id() -> str:
    """Return a compact timestamp for run folder naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def as_list(x: Any) -> List[Any]:
    """
    Convert Paddle/Numpy container outputs into a plain Python list without
    triggering ambiguous boolean checks.

    PaddleOCR often returns numpy.ndarray for fields like rec_boxes/rec_scores/rec_polys.
    """
    if x is None:
        return []
    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            pass
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]


def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert common non-JSON-serializable objects to JSON-friendly types.

    Handles:
    - pathlib.Path
    - numpy arrays / scalars
    - tuples/sets -> lists
    """
    if isinstance(obj, Path):
        return str(obj)

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def json_dump_safe(payload: Dict[str, Any], indent: int = 2) -> str:
    """
    JSON-dump a payload robustly.

    Primary path: dump after to_jsonable()
    Fallback: default=str to prevent rare non-serializable objects from killing the run.
    """
    try:
        return json.dumps(to_jsonable(payload), indent=indent)
    except TypeError:
        # Last resort: stringify unknown objects
        return json.dumps(to_jsonable(payload), indent=indent, default=str)


def normalize_text(s: Any) -> str:
    """
    Normalize OCR text for robust exact-match checks.

    Policy:
    - convert to string
    - uppercase
    - keep only alphanumerics (A-Z, 0-9)
    """
    if s is None:
        return ""
    s = str(s).strip().upper()
    return "".join(ch for ch in s if ch.isalnum())


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance (unit costs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def poly_area(poly: Sequence[Sequence[Union[int, float]]]) -> float:
    """Compute polygon area via shoelace formula."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(area) / 2.0


def poly_to_bbox(poly: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    """Convert polygon points to axis-aligned bbox (x1,y1,x2,y2)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def safe_get_first_result(ocr_predict_output: Any) -> Any:
    """For single-image crops, take first element if list-like."""
    if isinstance(ocr_predict_output, list) and len(ocr_predict_output) > 0:
        return ocr_predict_output[0]
    return ocr_predict_output


def ocr_result_to_dict(res_obj: Any) -> Dict[str, Any]:
    """
    Convert PaddleOCR result object into a plain dict.

    Supports:
    - dict
    - res_obj.json / res_obj.res
    - res_obj.__dict__
    Also unwraps {"res": {...}}.
    """
    data: Any = None

    if isinstance(res_obj, dict):
        data = res_obj
    else:
        for attr in ("json", "res"):
            if hasattr(res_obj, attr):
                try:
                    data = getattr(res_obj, attr)
                    if callable(data):
                        data = data()
                    break
                except Exception:
                    data = None
        if data is None:
            try:
                data = dict(res_obj.__dict__)
            except Exception:
                data = None

    if data is None:
        return {}

    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        data = data["res"]

    return data if isinstance(data, dict) else {}


def select_largest_prediction(ocr_dict: Dict[str, Any]) -> SelectedPrediction:
    """
    Select the OCR detection corresponding to the largest box.

    Preference order:
    1) Use rec_boxes (axis-aligned boxes) if present.
    2) Else compute area from rec_polys / dt_polys polygons.

    Returns idx=None when no boxes exist.
    """
    rec_texts = as_list(ocr_dict.get("rec_texts"))
    rec_scores = as_list(ocr_dict.get("rec_scores"))
    rec_boxes = as_list(ocr_dict.get("rec_boxes"))

    rec_polys = as_list(ocr_dict.get("rec_polys"))
    if not rec_polys:
        rec_polys = as_list(ocr_dict.get("dt_polys"))

    if max(len(rec_texts), len(rec_boxes), len(rec_polys)) == 0:
        return SelectedPrediction(idx=None, text="", score=None, bbox_xyxy=None, poly=None)

    areas: List[float] = []
    polys: List[Optional[List[List[int]]]] = []
    bboxes: List[Optional[Tuple[int, int, int, int]]] = []

    if len(rec_boxes) > 0:
        for i in range(len(rec_boxes)):
            box = rec_boxes[i]
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
                areas.append(max(0, x2 - x1) * max(0, y2 - y1))
                bboxes.append((x1, y1, x2, y2))
            except Exception:
                areas.append(0.0)
                bboxes.append(None)

            poly_i = None
            if i < len(rec_polys) and isinstance(rec_polys[i], list):
                try:
                    poly_i = [[int(p[0]), int(p[1])] for p in rec_polys[i]]
                except Exception:
                    poly_i = None
            polys.append(poly_i)
    else:
        for i in range(len(rec_polys)):
            poly = rec_polys[i]
            try:
                poly_int = [[int(p[0]), int(p[1])] for p in poly]
                areas.append(poly_area(poly_int))
                bboxes.append(poly_to_bbox(poly_int))
                polys.append(poly_int)
            except Exception:
                areas.append(0.0)
                bboxes.append(None)
                polys.append(None)

    best_idx = int(max(range(len(areas)), key=lambda i: areas[i])) if areas else None
    if best_idx is None:
        return SelectedPrediction(idx=None, text="", score=None, bbox_xyxy=None, poly=None)

    text = rec_texts[best_idx] if best_idx < len(rec_texts) else ""
    score = None
    if best_idx < len(rec_scores):
        try:
            score = float(rec_scores[best_idx])
        except Exception:
            score = None

    return SelectedPrediction(
        idx=best_idx,
        text=str(text),
        score=score,
        bbox_xyxy=bboxes[best_idx] if best_idx < len(bboxes) else None,
        poly=polys[best_idx] if best_idx < len(polys) else None,
    )


def draw_ocr_visualization(
    image_path: Path,
    ocr_dict: Dict[str, Any],
    selected: SelectedPrediction,
    out_path: Path,
) -> None:
    """
    Save a visualization image with OCR polygons/boxes drawn.

    - All boxes drawn in blue
    - Selected box drawn in red (thicker)
    - Labels include recognized text + score (if available)

    Note: This function never returns objects that are later serialized; it only writes the image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    rec_texts = as_list(ocr_dict.get("rec_texts"))
    rec_scores = as_list(ocr_dict.get("rec_scores"))

    polys = as_list(ocr_dict.get("rec_polys"))
    if not polys:
        polys = as_list(ocr_dict.get("dt_polys"))

    # Keep font local; never store in JSON payload.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    for i, poly in enumerate(polys):
        try:
            poly_pts = [(int(p[0]), int(p[1])) for p in poly]
        except Exception:
            continue

        is_sel = (selected.idx is not None and i == selected.idx)
        color = (255, 0, 0) if is_sel else (0, 128, 255)
        width = 3 if is_sel else 2

        draw.line(poly_pts + [poly_pts[0]], fill=color, width=width)

        x1, y1, _, _ = poly_to_bbox([[p[0], p[1]] for p in poly_pts])
        t = rec_texts[i] if i < len(rec_texts) else ""
        sc = rec_scores[i] if i < len(rec_scores) else None

        label = f"{t}"
        if sc is not None:
            try:
                label += f" ({float(sc):.2f})"
            except Exception:
                pass

        if label:
            pad = 2
            try:
                bbox = draw.textbbox((0, 0), label, font=font)  # type: ignore
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                # fallback rough sizing
                tw, th = (len(label) * 6, 11)

            draw.rectangle(
                [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1],
                fill=(255, 255, 255),
            )
            draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def build_file_mapping(meta_df: pd.DataFrame) -> Dict[str, str]:
    """Map crop filename -> full crop_path using meta.csv's crop_path column."""
    mapping: Dict[str, str] = {}
    if "crop_path" not in meta_df.columns:
        return mapping
    for p in meta_df["crop_path"].dropna().astype(str).tolist():
        mapping[Path(p).name] = p
    return mapping


def find_default_output_root() -> Path:
    """Default output directory: ./runs/ocr"""
    return Path.cwd() / "runs" / "ocr"


# ----------------------------- Main evaluation ----------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR on crop dataset; select prediction from largest OCR box."
    )
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels.csv (ground truth).")
    parser.add_argument("--meta_csv", type=str, required=True, help="Path to meta.csv (contains crop_path).")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(find_default_output_root()),
        help="Output root directory (default: ./runs/ocr).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"ppocr_eval_{now_run_id()}",
        help="Run directory name under output_root.",
    )
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="PaddleOCR device.")
    parser.add_argument("--lang", type=str, default="en", help="OCR language (default: en).")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0 = all).")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first exception.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels_csv = Path(args.labels_csv)
    meta_csv = Path(args.meta_csv)
    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name

    pred_json_dir = run_dir / "pred_json"
    pred_img_dir = run_dir / "pred_imgs"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_json_dir.mkdir(parents=True, exist_ok=True)
    pred_img_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv)
    meta_df = pd.read_csv(meta_csv)

    required_cols = {"file_name", "bib_id"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")

    file_map = build_file_mapping(meta_df)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
        device=args.device,
    )

    rows_out: List[Dict[str, Any]] = []
    n_total = len(labels_df)
    n_eval = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)

    t0 = time.time()
    n_with_boxes = 0
    n_no_boxes = 0
    n_errors = 0

    for idx_row in range(n_eval):
        row = labels_df.iloc[idx_row].to_dict()
        file_name = str(row["file_name"])
        gt = str(row["bib_id"])

        crop_path_str = file_map.get(file_name, "")
        crop_path = Path(crop_path_str) if crop_path_str else None

        out_row: Dict[str, Any] = dict(row)
        out_row["crop_path"] = str(crop_path) if crop_path else ""

        stem = Path(file_name).stem
        json_out_path = pred_json_dir / f"{stem}.json"
        img_out_path = pred_img_dir / f"{stem}.jpg"
        out_row["pred_json_path"] = str(json_out_path)
        out_row["pred_img_path"] = str(img_out_path)

        # Defaults
        out_row["pred_text"] = ""
        out_row["pred_text_norm"] = ""
        out_row["pred_score"] = None
        out_row["pred_bbox_x1"] = None
        out_row["pred_bbox_y1"] = None
        out_row["pred_bbox_x2"] = None
        out_row["pred_bbox_y2"] = None
        out_row["n_boxes"] = 0
        out_row["error"] = ""
        out_row["viz_error"] = ""

        # Define payload up-front so it always exists (fixes your NameError)
        payload: Dict[str, Any] = {
            "file_name": file_name,
            "crop_path": str(crop_path) if crop_path else "",
            "ground_truth": {"bib_id": gt, "bib_id_norm": normalize_text(gt)},
        }

        if crop_path is None or not crop_path.exists():
            msg = f"Missing crop_path for {file_name} (looked up via meta.csv mapping)."
            out_row["error"] = msg
            payload["error"] = msg
            n_errors += 1

            # Still write JSON for debugging
            json_out_path.parent.mkdir(parents=True, exist_ok=True)
            json_out_path.write_text(json_dump_safe(payload, indent=2), encoding="utf-8")

            rows_out.append(out_row)
            if args.fail_fast:
                raise FileNotFoundError(msg)
            continue

        try:
            pred = ocr.predict(str(crop_path))
            res0 = safe_get_first_result(pred)
            ocr_dict = ocr_result_to_dict(res0)

            selected = select_largest_prediction(ocr_dict)

            rec_texts = as_list(ocr_dict.get("rec_texts"))
            out_row["n_boxes"] = int(len(rec_texts))

            if selected.idx is None:
                n_no_boxes += 1
            else:
                n_with_boxes += 1

            out_row["pred_text"] = selected.text
            out_row["pred_text_norm"] = normalize_text(selected.text)
            out_row["pred_score"] = selected.score

            if selected.bbox_xyxy is not None:
                x1, y1, x2, y2 = selected.bbox_xyxy
                out_row["pred_bbox_x1"] = x1
                out_row["pred_bbox_y1"] = y1
                out_row["pred_bbox_x2"] = x2
                out_row["pred_bbox_y2"] = y2

            payload["ocr_output"] = ocr_dict
            payload["selected"] = {
                "idx": selected.idx,
                "text": selected.text,
                "text_norm": normalize_text(selected.text),
                "score": selected.score,
                "bbox_xyxy": selected.bbox_xyxy,
                "poly": selected.poly,
            }

            # --- Write JSON (robust) ---
            json_out_path.parent.mkdir(parents=True, exist_ok=True)
            json_out_path.write_text(json_dump_safe(payload, indent=2), encoding="utf-8")

            # --- Write visualization (best effort, should not fail the whole sample) ---
            try:
                draw_ocr_visualization(
                    image_path=crop_path,
                    ocr_dict=ocr_dict,
                    selected=selected,
                    out_path=img_out_path,
                )
            except Exception as viz_e:
                out_row["viz_error"] = f"{type(viz_e).__name__}: {viz_e}"

        except Exception as e:
            out_row["error"] = f"{type(e).__name__}: {e}"
            payload["error"] = out_row["error"]
            n_errors += 1

            # Try to still write a JSON so you can inspect failures
            try:
                json_out_path.parent.mkdir(parents=True, exist_ok=True)
                json_out_path.write_text(json_dump_safe(payload, indent=2), encoding="utf-8")
            except Exception:
                pass

            if args.fail_fast:
                raise
        finally:
            rows_out.append(out_row)

    elapsed = time.time() - t0

    pred_df = pd.DataFrame(rows_out)

    gt_norm = pred_df["bib_id"].astype(str).map(normalize_text)
    pred_norm = pred_df["pred_text_norm"].astype(str).fillna("")
    has_pred = pred_df["n_boxes"].fillna(0).astype(int) > 0

    exact_norm = (pred_norm == gt_norm) & has_pred
    acc_norm = float(exact_norm.sum()) / float(len(pred_df)) if len(pred_df) else 0.0

    dists: List[int] = []
    for p, g, ok in zip(pred_norm.tolist(), gt_norm.tolist(), has_pred.tolist()):
        dists.append(len(g) if not ok else levenshtein(p, g))
    pred_df["edit_distance_norm"] = dists
    pred_df["is_exact_norm"] = exact_norm.astype(bool)

    mean_dist = float(sum(dists) / len(dists)) if dists else 0.0
    mean_cer = float(
        sum(d / max(1, len(g)) for d, g in zip(dists, gt_norm.tolist())) / len(dists)
    ) if dists else 0.0

    summary = {
        "run_name": args.run_name,
        "labels_csv": str(labels_csv),
        "meta_csv": str(meta_csv),
        "output_dir": str(run_dir),
        "device": args.device,
        "lang": args.lang,
        "n_samples": int(len(pred_df)),
        "n_with_boxes": int(n_with_boxes),
        "n_no_boxes": int(n_no_boxes),
        "n_errors": int(n_errors),
        "coverage": float(n_with_boxes / len(pred_df)) if len(pred_df) else 0.0,
        "accuracy_exact_norm": acc_norm,
        "mean_edit_distance_norm": mean_dist,
        "mean_cer_norm": mean_cer,
        "elapsed_sec": elapsed,
        "sec_per_sample": float(elapsed / len(pred_df)) if len(pred_df) else None,
    }

    pred_csv_path = run_dir / "predictions.csv"
    summary_path = run_dir / "prediction_summary.json"
    pred_df.to_csv(pred_csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== OCR Evaluation Summary ===")
    print(f"Run dir:              {run_dir}")
    print(f"Samples:              {summary['n_samples']}")
    print(f"Coverage (has boxes): {summary['coverage']:.3f} ({summary['n_with_boxes']}/{summary['n_samples']})")
    print(f"Errors:               {summary['n_errors']}")
    print(f"Accuracy (norm exact):{summary['accuracy_exact_norm']:.3f}")
    print(f"Mean edit dist (norm):{summary['mean_edit_distance_norm']:.3f}")
    print(f"Mean CER (norm):      {summary['mean_cer_norm']:.3f}")
    print(f"Time:                 {summary['elapsed_sec']:.2f}s ({summary['sec_per_sample']:.3f}s/sample)")

    mismatches = pred_df[~pred_df["is_exact_norm"]].copy()
    mismatches = mismatches[mismatches["error"].astype(str).str.len() == 0]
    if len(mismatches) > 0:
        mismatches = mismatches.sort_values("edit_distance_norm", ascending=False).head(10)
        print("\nTop mismatches (by edit distance):")
        for _, r in mismatches.iterrows():
            print(f"- {r['file_name']}: gt={r['bib_id']} pred={r['pred_text']} dist={r['edit_distance_norm']}")


if __name__ == "__main__":
    main()
