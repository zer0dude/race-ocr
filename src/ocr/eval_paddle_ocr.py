#!/usr/bin/env python3
"""
Evaluate PaddleOCR (PP-OCRv5 via paddleocr.PaddleOCR) on a crop dataset.

Key evaluation behavior (tailored for race-ocr):
- Each crop corresponds to exactly one label string (e.g., bib_id), OR can be "empty".
- OCR may return 0..N boxes; we select the prediction from the *largest* box.
- Bounding-box overlap with label boxes is NOT evaluated. Only string match matters.
- Confidence thresholding is supported:
    - Only predictions with score >= --conf_thresh are considered "valid predictions".
    - Below-threshold predictions are treated as "no prediction" for evaluation.
- Some images may be empty; for those, "no prediction" is considered correct.

Input files:
- labels.csv must contain at least: file_name, bib_id
  - If bib_id is empty/NaN/whitespace, it is treated as GT-empty.
  - labels.csv may optionally contain crop_path.
- meta.csv is optional; used only to map filename -> crop_path if provided.
- crops_dir can be used instead of meta.csv: crops_dir / file_name must exist.

Input path resolution (priority order):
1) If labels.csv provides crop_path and the file exists -> use it.
2) Else if --crops_dir is given -> use crops_dir / file_name.
3) Else if --meta_csv is given -> map basename -> crop_path from meta.csv.
4) Else -> fail with a clear error.

Outputs under:
    <output_root>/<run_name>/

Artifacts:
- pred_json/<stem>.json         Full OCR output + chosen box/text (+ GT + errors if any)
- pred_imgs/<stem>.jpg          Visualization (all boxes + chosen highlighted) (best-effort)
- predictions.csv               labels.csv extended with prediction fields + errors
- prediction_summary.json       Metrics + config

Metrics:
- coverage_valid: fraction of samples with a valid prediction (passes confidence threshold)
- accuracy_allow_empty: fraction of samples correct under:
    - GT non-empty: valid pred AND pred_norm==gt_norm
    - GT empty: no valid pred
- precision_valid: among valid predictions only, fraction with pred_norm==gt_norm

Example:
python src/ocr/eval_paddle_ocr.py \
  --labels_csv /data/repos/race-ocr/data/ocr/handwritten_256/labels.csv \
  --crops_dir /data/repos/race-ocr/data/ocr/handwritten_256/bib_crops \
  --device gpu \
  --run_name handwritten_256
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
        return json.dumps(to_jsonable(payload), indent=indent, default=str)


def normalize_text(s: Any) -> str:
    """
    Normalize OCR / label text for robust matching.

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
    Select the OCR detection corresponding to the largest box/polygon area.
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


def draw_ocr_visualization(image_path: Path, ocr_dict: Dict[str, Any], selected: SelectedPrediction, out_path: Path) -> None:
    """Save visualization image with polygons, highlighting the selected prediction."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    rec_texts = as_list(ocr_dict.get("rec_texts"))
    rec_scores = as_list(ocr_dict.get("rec_scores"))

    polys = as_list(ocr_dict.get("rec_polys"))
    if not polys:
        polys = as_list(ocr_dict.get("dt_polys"))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    for i, poly in enumerate(polys):
        try:
            pts = [(int(p[0]), int(p[1])) for p in poly]
        except Exception:
            continue

        is_sel = (selected.idx is not None and i == selected.idx)
        color = (255, 0, 0) if is_sel else (0, 128, 255)
        width = 3 if is_sel else 2
        draw.line(pts + [pts[0]], fill=color, width=width)

        x1, y1, _, _ = poly_to_bbox([[p[0], p[1]] for p in pts])
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
                tw, th = (len(label) * 6, 11)

            draw.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1], fill=(255, 255, 255))
            draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def build_file_mapping_from_meta(meta_df: pd.DataFrame) -> Dict[str, str]:
    """Map crop basename -> full crop_path using meta.csv crop_path."""
    mapping: Dict[str, str] = {}
    if "crop_path" not in meta_df.columns:
        return mapping
    for p in meta_df["crop_path"].dropna().astype(str).tolist():
        mapping[Path(p).name] = p
    return mapping


def find_default_output_root() -> Path:
    """Default output directory: ./runs/ocr"""
    return Path.cwd() / "runs" / "ocr"


def resolve_crop_path(row: Dict[str, Any], crops_dir: Optional[Path], meta_map: Optional[Dict[str, str]]) -> Tuple[Optional[Path], str]:
    """Resolve crop path. See module docstring for priority order."""
    file_name = str(row.get("file_name", ""))

    if "crop_path" in row and str(row["crop_path"]).strip():
        p = Path(str(row["crop_path"]))
        if p.exists():
            return p, "labels.crop_path"
        return None, f"labels.crop_path missing on disk: {p}"

    if crops_dir is not None:
        p = crops_dir / file_name
        if p.exists():
            return p, "crops_dir/file_name"
        return None, f"crops_dir join missing on disk: {p}"

    if meta_map is not None:
        p_str = meta_map.get(file_name, "")
        if p_str:
            p = Path(p_str)
            if p.exists():
                return p, "meta.csv mapping"
            return None, f"meta.csv mapped path missing on disk: {p}"
        return None, f"meta.csv has no entry for file_name={file_name}"

    return None, "No path source available (need labels.crop_path or --crops_dir or --meta_csv)"


# ----------------------------- CLI ----------------------------- #

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """
    Add a pair of boolean flags:
      --<name> / --no-<name>
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=f"Enable {help_text}")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR on crop dataset; select prediction from largest OCR box."
    )
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels.csv (ground truth).")
    parser.add_argument("--meta_csv", type=str, default=None,
                        help="Optional path to meta.csv (contains crop_path). Not needed if --crops_dir is provided.")
    parser.add_argument("--crops_dir", type=str, default=None,
                        help="Directory containing crop images referenced by labels.csv file_name.")
    parser.add_argument("--output_root", type=str, default=str(find_default_output_root()),
                        help="Output root directory (default: ./runs/ocr).")
    parser.add_argument("--run_name", type=str, default=f"ppocr_eval_{now_run_id()}",
                        help="Run directory name under output_root.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="PaddleOCR device.")
    parser.add_argument("--lang", type=str, default="en", help="OCR language (default: en).")

    # New: confidence threshold
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.75,
        help="Only predictions with selected-box score >= this threshold are treated as valid predictions.",
    )

    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0 = all).")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first exception.")

    add_bool_flag(parser, "use_doc_orientation_classify", default=False,
                  help_text="document orientation classification model")
    add_bool_flag(parser, "use_doc_unwarping", default=False,
                  help_text="document unwarping / rectification model")
    add_bool_flag(parser, "use_textline_orientation", default=False,
                  help_text="textline orientation classification model")

    return parser.parse_args()


# ----------------------------- Main ----------------------------- #

def main() -> None:
    args = parse_args()

    labels_csv = Path(args.labels_csv)
    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name

    pred_json_dir = run_dir / "pred_json"
    pred_img_dir = run_dir / "pred_imgs"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_json_dir.mkdir(parents=True, exist_ok=True)
    pred_img_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv)
    required_cols = {"file_name", "bib_id"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir is not None and not crops_dir.exists():
        raise FileNotFoundError(f"--crops_dir does not exist: {crops_dir}")

    meta_map: Optional[Dict[str, str]] = None
    if args.meta_csv:
        meta_df = pd.read_csv(Path(args.meta_csv))
        meta_map = build_file_mapping_from_meta(meta_df)

    ocr = PaddleOCR(
        use_doc_orientation_classify=bool(args.use_doc_orientation_classify),
        use_doc_unwarping=bool(args.use_doc_unwarping),
        use_textline_orientation=bool(args.use_textline_orientation),
        lang=args.lang,
        device=args.device,
    )

    rows_out: List[Dict[str, Any]] = []
    n_total = len(labels_df)
    n_eval = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)

    t0 = time.time()
    n_errors = 0

    for idx_row in range(n_eval):
        row = labels_df.iloc[idx_row].to_dict()
        file_name = str(row["file_name"])
        gt_raw = row.get("bib_id", "")
        gt_norm = normalize_text(gt_raw)

        crop_path, path_source = resolve_crop_path(row, crops_dir=crops_dir, meta_map=meta_map)

        out_row: Dict[str, Any] = dict(row)
        out_row["crop_path"] = str(crop_path) if crop_path else ""
        out_row["path_source"] = path_source

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
        out_row["pred_is_valid"] = False  # NEW: passes confidence threshold
        out_row["gt_is_empty"] = (gt_norm == "")  # NEW
        out_row["error"] = ""
        out_row["viz_error"] = ""

        payload: Dict[str, Any] = {
            "file_name": file_name,
            "crop_path": str(crop_path) if crop_path else "",
            "path_source": path_source,
            "ground_truth": {"raw": str(gt_raw), "norm": gt_norm, "is_empty": (gt_norm == "")},
            "config": {
                "device": args.device,
                "lang": args.lang,
                "conf_thresh": args.conf_thresh,
                "use_doc_orientation_classify": bool(args.use_doc_orientation_classify),
                "use_doc_unwarping": bool(args.use_doc_unwarping),
                "use_textline_orientation": bool(args.use_textline_orientation),
            },
        }

        if crop_path is None or not crop_path.exists():
            msg = f"Missing crop_path: {path_source}"
            out_row["error"] = msg
            payload["error"] = msg
            n_errors += 1
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

            # Raw selected values
            out_row["pred_text"] = selected.text
            out_row["pred_text_norm"] = normalize_text(selected.text)
            out_row["pred_score"] = selected.score

            if selected.bbox_xyxy is not None:
                x1, y1, x2, y2 = selected.bbox_xyxy
                out_row["pred_bbox_x1"] = x1
                out_row["pred_bbox_y1"] = y1
                out_row["pred_bbox_x2"] = x2
                out_row["pred_bbox_y2"] = y2

            # NEW: apply confidence threshold to decide whether this is a "valid prediction"
            score_ok = (selected.score is not None and float(selected.score) >= float(args.conf_thresh))
            has_any_box = out_row["n_boxes"] > 0
            pred_is_valid = bool(has_any_box and score_ok)
            out_row["pred_is_valid"] = pred_is_valid

            payload["ocr_output"] = ocr_dict
            payload["selected"] = {
                "idx": selected.idx,
                "text": selected.text,
                "text_norm": out_row["pred_text_norm"],
                "score": selected.score,
                "bbox_xyxy": selected.bbox_xyxy,
                "poly": selected.poly,
                "pred_is_valid": pred_is_valid,
                "conf_thresh": args.conf_thresh,
            }

            # Save JSON
            json_out_path.write_text(json_dump_safe(payload, indent=2), encoding="utf-8")

            # Save viz best-effort
            try:
                draw_ocr_visualization(crop_path, ocr_dict, selected, img_out_path)
            except Exception as viz_e:
                out_row["viz_error"] = f"{type(viz_e).__name__}: {viz_e}"

        except Exception as e:
            out_row["error"] = f"{type(e).__name__}: {e}"
            payload["error"] = out_row["error"]
            n_errors += 1
            try:
                json_out_path.write_text(json_dump_safe(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
            if args.fail_fast:
                raise
        finally:
            rows_out.append(out_row)

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(rows_out)

    # ----------------------------- Metrics ----------------------------- #
    gt_norm_series = pred_df["bib_id"].map(normalize_text)
    gt_is_empty = (gt_norm_series == "")

    pred_norm_series = pred_df["pred_text_norm"].astype(str).fillna("")
    pred_score = pred_df["pred_score"]
    n_boxes = pred_df["n_boxes"].fillna(0).astype(int)

    # "valid prediction" = has boxes AND score >= conf_thresh
    score_ok = pred_score.apply(lambda x: (x is not None) and (float(x) >= float(args.conf_thresh)))
    pred_is_valid = (n_boxes > 0) & score_ok

    # Precision-style metric among valid predictions only
    valid_matches = pred_is_valid & (pred_norm_series == gt_norm_series)
    precision_valid = float(valid_matches.sum()) / float(pred_is_valid.sum()) if int(pred_is_valid.sum()) > 0 else 0.0

    # Accuracy allowing empty GT:
    # - GT non-empty: must have valid prediction and match
    # - GT empty: must have NO valid prediction
    correct_nonempty = (~gt_is_empty) & pred_is_valid & (pred_norm_series == gt_norm_series)
    correct_empty = gt_is_empty & (~pred_is_valid)
    accuracy_allow_empty = float((correct_nonempty | correct_empty).sum()) / float(len(pred_df)) if len(pred_df) else 0.0

    coverage_valid = float(pred_is_valid.sum()) / float(len(pred_df)) if len(pred_df) else 0.0

    # Keep your distance metrics too (computed on normalized strings; for empty GT it's fine)
    dists: List[int] = []
    for p, g in zip(pred_norm_series.tolist(), gt_norm_series.tolist()):
        dists.append(levenshtein(p, g))
    pred_df["edit_distance_norm"] = dists

    # Report booleans used above
    pred_df["gt_is_empty"] = gt_is_empty.astype(bool)
    pred_df["pred_is_valid"] = pred_is_valid.astype(bool)
    pred_df["is_match_norm"] = (pred_norm_series == gt_norm_series).astype(bool)

    summary = {
        "run_name": args.run_name,
        "labels_csv": str(labels_csv),
        "meta_csv": str(args.meta_csv) if args.meta_csv else None,
        "crops_dir": str(crops_dir) if crops_dir else None,
        "output_dir": str(run_dir),
        "config": {
            "device": args.device,
            "lang": args.lang,
            "conf_thresh": float(args.conf_thresh),
            "use_doc_orientation_classify": bool(args.use_doc_orientation_classify),
            "use_doc_unwarping": bool(args.use_doc_unwarping),
            "use_textline_orientation": bool(args.use_textline_orientation),
        },
        "n_samples": int(len(pred_df)),
        "n_errors": int(n_errors),
        "coverage_valid": coverage_valid,
        "accuracy_allow_empty": accuracy_allow_empty,
        "precision_valid": precision_valid,
        "elapsed_sec": elapsed,
        "sec_per_sample": float(elapsed / len(pred_df)) if len(pred_df) else None,
    }

    pred_csv_path = run_dir / "predictions.csv"
    summary_path = run_dir / "prediction_summary.json"
    pred_df.to_csv(pred_csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== OCR Evaluation Summary ===")
    print(f"Run dir:                 {run_dir}")
    print(f"Samples:                 {summary['n_samples']}")
    print(f"Errors:                  {summary['n_errors']}")
    print(f"Conf thresh:             {summary['config']['conf_thresh']:.2f}")
    print(f"Coverage (valid preds):  {summary['coverage_valid']:.3f} ({int(pred_is_valid.sum())}/{summary['n_samples']})")
    print(f"Accuracy (allow empty):  {summary['accuracy_allow_empty']:.3f}")
    print(f"Precision (valid preds): {summary['precision_valid']:.3f}")
    print(f"Time:                    {summary['elapsed_sec']:.2f}s ({summary['sec_per_sample']:.3f}s/sample)")

    # Optional: show top mismatches among valid predictions
    mism = pred_df[pred_is_valid & (pred_df["is_match_norm"] == False) & (pred_df["error"].astype(str).str.len() == 0)].copy()
    if len(mism) > 0:
        mism = mism.assign(gt_norm=gt_norm_series, pred_norm=pred_norm_series)
        mism["dist"] = mism.apply(lambda r: levenshtein(str(r["pred_norm"]), str(r["gt_norm"])), axis=1)
        mism = mism.sort_values("dist", ascending=False).head(10)
        print("\nTop mismatches among valid predictions:")
        for _, r in mism.iterrows():
            print(f"- {r['file_name']}: gt={r['bib_id']} pred={r['pred_text']} score={r['pred_score']} dist={r['dist']}")


if __name__ == "__main__":
    main()
