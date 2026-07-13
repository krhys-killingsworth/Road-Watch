# RoadWatch — Automated Road Damage Detector

**Course:** ITAI 1378 — Computer Vision · Houston Community College
**Team:** Krhys Killingsworth, Nnaemeka Ofoegbu
**Project Tier:** Tier 2 — fine-tune a pretrained detector on a domain-specific dataset

> RoadWatch detects and classifies road damage (cracks and potholes) from street-level
> images using a YOLO11 object detector fine-tuned on the public RDD2022 dataset.

---

## Tier Selection & Justification

**Tier 2.** This project takes a COCO-pretrained model (YOLO11) and fine-tunes it on a new,
domain-specific dataset (RDD2022) with a custom 5-class damage label set. It goes beyond running an
off-the-shelf model on its original classes (Tier 1) but does not train a novel architecture
from scratch (Tier 3). Transfer learning is the right scope for a 6-week solo midterm: it
delivers a working, measurable detector on free compute.

---

## Problem Statement

Road agencies still inspect pavement manually or with slow specialized vehicles, so damage is
logged late, and repair cost multiplies the longer a defect sits. In flood-prone Houston,
water accelerates the crack-to-pothole cycle, so frequent, low-cost detection has direct safety
and budget payoff for drivers, cyclists, and public-works teams.

## Solution Overview

A phone or dashcam captures road images; a fine-tuned YOLO11 detector locates and labels each
defect; the system outputs annotated images plus a per-image damage-count/severity report (CSV).
The output is designed to be usable by a non-technical public-works team — annotated images and a
spreadsheet ready to map.

---

## Technical Approach

| Component | Choice | Why |
|---|---|---|
| **Technique** | Object detection | Damage must be *located and counted*, not just classified. |
| **Model** | Ultralytics **YOLO11s** | Fast, proven, well-documented; the `s` variant fine-tunes on a free GPU. Upgrade path: YOLO11m. |
| **Framework** | PyTorch + Ultralytics | One-line train/val/export API, built-in augmentation, easy ONNX export for the demo. |

YOLO11 ships with COCO-pretrained weights, so the backbone already understands edges and
textures; fine-tuning on road imagery converges in hours on one GPU. *(YOLO26, released Jan 2026,
is a possible newer swap, but YOLO11 is the stable, documented choice for this timeframe.)*

---

## Dataset (actual — updated after Week 1–2)

| Field | Detail |
|---|---|
| **Source used** | RDD2022 mirror on Roboflow Universe (`road-condition-detection-zr4jv/rdd2022-22jrg`, v1), downloaded in YOLO format via the Roboflow SDK. |
| **Size (actual)** | ~18.8k train / 3,921 val images. |
| **Labels (5)** | The export shipped unnamed classes (`0–4`); identified by visual audit of labeled samples: `longitudinal_crack` (D00), `transverse_crack` (D10), `alligator_crack` (D20), `pothole` (D40), `repair` (patched asphalt). |
| **Preparation** | No format conversion needed (YOLO-native export). Class names written into `data.yaml` after the visual audit. Known imbalance: `repair` is the rare class (~965 val instances vs. ~3,890 for longitudinal). |

Original dataset paper: Arya et al., *RDD2022: A multi-national image dataset for automatic Road
Damage Detection*, arXiv:2209.08538 (47,420 images / 6 countries in the full release; official
copy on [Figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)).
The VOC→YOLO conversion path (`src/prepare_data.py`) is kept for anyone using the official release.

---

## Success Metrics

| Type | Metric | Target | **Baseline result (5 epochs)** |
|---|---|---|---|
| **Primary** | mAP@0.5 (all classes) | **≥ 0.55** | 0.452 — on track, curve still climbing |
| **Primary** | Pothole recall | **≥ 0.60** | **0.646 ✅ met** (pothole AP50 0.634) |
| **Secondary** | Inference speed | **< 50 ms/image** | **8.8 ms/img ✅ met** (T4) |

*Realistic, not inflated:* detection mAP is stricter than classification accuracy. The
CRDDC'2022 winner reached ~F1 0.77 across all six countries, so mAP@0.5 ≥ 0.55 on a scoped
subset is honest and achievable. Per-class AP, precision, and recall are all reported.

### Baseline training run (Week 2 milestone — complete)

YOLO11s fine-tuned for 5 epochs on a Tesla T4 (Colab), `batch=8, imgsz=640` — 0.66 hrs:

| Class | Val instances | Precision | Recall | AP@0.5 |
|---|---|---|---|---|
| all | 9,740 | 0.518 | 0.448 | **0.452** |
| longitudinal_crack | 3,890 | 0.499 | 0.448 | 0.422 |
| transverse_crack | 1,769 | 0.500 | 0.360 | 0.381 |
| alligator_crack | 1,553 | 0.633 | 0.429 | 0.507 |
| pothole | 1,563 | 0.583 | **0.646** | 0.634 |
| repair | 965 | 0.376 | 0.357 | 0.316 |

mAP rose every epoch (0.209 → 0.452) with no plateau — the full 50-epoch Week 3 run with
augmentation is expected to clear the 0.55 target. `repair` is the weakest class (rare +
visually subtle), the focus of Week 3 rebalancing.

---

## Week-by-Week Plan

| Week | Focus | Tasks | Milestone |
|---|---|---|---|
| 1 ✅ | Setup & data | Download RDD2022 (Roboflow mirror), verify & name classes, EDA | **Dataset ready — done** |
| 2 ✅ | Baseline train | Fine-tune YOLO11s on T4, log first mAP | **Model working — done (mAP@0.5 0.452)** |
| 3 | Improve | 50-epoch run + augmentation, rebalance `repair` class, tune / try YOLO11m | Accuracy up (target ≥ 0.55) |
| 4 | Demo build | Inference on local road clips, annotated video + CSV | Demo ready |
| 5 | Finalize | Error analysis, README, docs, final eval | Everything done |
| 6 | Present | Slides, dry run, deliver in class | 🎉 Presentation day |

---

## Resources Needed

| Resource | Notes |
|---|---|
| **Compute** | Google Colab (free T4) — primary. Kaggle (30 GPU-hrs/wk) — backup. |
| **Frameworks / APIs** | PyTorch, Ultralytics YOLO11, Roboflow (format conversion), OpenCV, pandas. |
| **Data** | RDD2022 — free public dataset. |
| **Estimated cost** | **$0.** Optional Colab Pro (~$10/mo) only if free GPU is too limiting. |

---

## Risks & Mitigation

| Risk | Probability | Status / Mitigation (Plan B) |
|---|---|---|
| Class imbalance (`repair` rare, weakest AP) | Medium | **Confirmed in baseline.** Week 3: oversample + heavy augmentation, class weights; fallback: merge into an "other" class. |
| mAP below 0.55 after full run | Medium | Upgrade YOLO11s → 11m; more epochs; mosaic augmentation; or narrow to fewer classes. |
| Compute limits (Colab timeout / RAM) | Medium | **Hit & solved in Week 2:** GPU runtime + `batch=8, workers=2, cache=False`. Kaggle (30 GPU-hrs/wk) as backup; checkpoint & resume for long runs. |
| Annotation format mismatch | ~~Low~~ | **Resolved:** Roboflow mirror ships YOLO-native labels. VOC converter kept in `src/` for the official release path. |
| Unnamed classes in mirror export | — | **Hit & solved:** export shipped classes `0–4`; identified by visual audit of labeled crops and written into `data.yaml`. |
| Domain gap (global data ≠ Houston roads) | Low | Collect a small local phone-photo test set for qualitative checks (Week 4). |

---

## Repository Structure

```
ITAI 1378 Midterm_RoadDamageDetector/
├── README.md                 ← this file
├── requirements.txt          ← Python packages
├── AI_USAGE_LOG.md           ← record of AI-tool assistance
├── configs/
│   └── data.yaml             ← YOLO dataset config (paths + class names)
├── models/
│   ├── yolo11.yaml           ← YOLO11 architecture (all scales)
│   ├── download_weights.py   ← fetch pretrained yolo11n/s/m.pt
│   └── README.md             ← model & weights notes
├── src/
│   ├── prepare_data.py       ← RDD2022 (VOC XML) → YOLO-ready splits + data.yaml
│   ├── voc_to_yolo.py        ← VOC XML → YOLO txt converter (used by prepare_data)
│   ├── train.py              ← fine-tune YOLO11
│   ├── evaluate.py           ← mAP + per-class metrics
│   └── detect.py             ← inference → annotated images + damage-count CSV
├── notebooks/
│   └── 01_exploration.ipynb  ← EDA + training/inference walkthrough
├── data/
│   └── README.md             ← dataset download & prep instructions
└── docs/
    └── proposal.pdf          ← presentation slides
```

## Quickstart

```bash
pip install -r requirements.txt

# Data — Option B (used in this project): Roboflow Universe mirror, YOLO-native
#   see notebooks/01_exploration.ipynb cell 1 (needs a free Roboflow API key)
# Data — Option A: official Figshare release (VOC XML), then:
#   python src/prepare_data.py --country_dirs data/raw/United_States data/raw/Japan --out data

# train  →  evaluate  →  detect
python src/train.py    --data configs/data.yaml --model yolo11s.pt --epochs 50 --aug
python src/evaluate.py --weights runs/detect/roadwatch/weights/best.pt --data configs/data.yaml
python src/detect.py   --weights runs/detect/roadwatch/weights/best.pt --source demo_images/
```

## Pipeline status

**Week 1–2 complete.** The full pipeline has been run for real on Colab (T4 GPU): dataset
downloaded and audited, YOLO11s fine-tuned for 5 epochs (~19k train images), evaluated
(mAP@0.5 = 0.452; pothole recall 0.646 — recall target already met), and inference demo
executed on validation images. The executed notebook with all outputs is
`notebooks/01_exploration.ipynb`. Next: the 50-epoch Week 3 run with augmentation.

> **Weights note:** pretrained `yolo11*.pt` are not committed (large binaries). They auto-download
> on first use, or run `python models/download_weights.py`. The trained baseline checkpoint
> (`best.pt`, 19 MB) is kept out of git for the same reason.
