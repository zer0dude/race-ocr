# race-ocr

Multi-stage computer vision pipeline for race photos (marathons, cycling, etc.):

1) **Detection (YOLO):** detect bib/headband regions  
2) **Classification (future):** refine/route detections  
3) **OCR (PaddleOCR):** read bib IDs from crops

This repo uses **separate Python virtual environments** to avoid GPU/CUDA binary conflicts:
- `.venv-yolo` for detection (+ future classifiers, PyTorch)
- `.venv-paddle` for OCR (PaddlePaddle / PaddleOCR)

---

## Quickstart

### Prerequisites (AWS EC2)
- Ubuntu
- NVIDIA driver installed (`nvidia-smi` works)
- Python 3.12 available (`python3.12 --version`)

---

## YOLO environment (detection)

### Create / install
```bash
bash scripts/setup_yolo_env.sh
```

### Activate
```bash
source .venv-yolo/bin/activate
```

### Smoke test
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
yolo --version
```

### Example inference
```bash
yolo predict \
  model=runs/job9_y11s_960_base/weights/best.pt \
  source=/data/repos/race-ocr/data/raw/job-9_v1/images/train/FAJ_6639.jpg \
  imgsz=960 conf=0.25 device=0
```

Deactivate:
```bash
deactivate
```

---

## OCR environment (PaddleOCR)
> Setup instructions coming next (see `scripts/setup_paddle_env.sh`).

---

## Repo structure (high level)

- `src/` — code
- `data/`
  - `raw/` — raw images + labels
  - `splits/` — train/val lists + YAML
  - `golden/` — small curated sets for evaluation
- `runs/` — model outputs (typically not committed)

---

## Reproducibility

We maintain two file types per environment:
- `requirements-<env>.in` — direct dependencies
- `requirements-<env>.txt` — frozen snapshot (`pip freeze` after a known-good install)

GPU frameworks (Torch / PaddlePaddle) are installed by the setup scripts to ensure correct CUDA builds.
