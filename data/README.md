# Dataset — RDD2022 (Roboflow mirror)

This folder holds the dataset locally. **The images are not committed to git** (too large).

## Source actually used (Week 1)

Roboflow Universe mirror: **`road-condition-detection-zr4jv/rdd2022-22jrg`** (version 1),
downloaded in YOLO format via the Roboflow SDK — see the download cell in
`notebooks/01_exploration.ipynb` (requires a free Roboflow API key; store it as a Colab
secret, never hardcoded).

- **Size:** ~18.8k train / 3,921 val images
- **Layout:** Roboflow-style `train/images`, `train/labels`, `valid/images`, `valid/labels` + `data.yaml`
- **Note:** the export shipped **unnamed classes (`0–4`)**; they were identified by visual audit
  of labeled crops and written into `data.yaml` (see notebook). Re-run that rename cell after any
  fresh download.

## Classes used (5)

| Id | Class | RDD code | Notes |
|----|-------|----------|-------|
| 0 | longitudinal_crack | D00 | most common (~3.9k val instances) |
| 1 | transverse_crack | D10 | |
| 2 | alligator_crack | D20 | |
| 3 | pothole | D40 | **primary metric class** — recall target ≥ 0.60 |
| 4 | repair | — | patched asphalt; rarest & weakest class (imbalance focus) |

## About the original RDD2022

- 47,420 road images from 6 countries (Japan, India, Czech Republic, Norway, USA, China), 55,000+ annotated instances, Pascal VOC XML boxes; free for research/education.
- **Paper:** Arya et al., *RDD2022: A multi-national image dataset for automatic Road Damage Detection*, arXiv:2209.08538.
- **Official copy (Option A):** https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547
  → if using this instead, run `src/prepare_data.py` to convert VOC XML → YOLO and build splits.

## Expected layout after download (Roboflow export)

```
data/                  (or wherever the SDK downloads, e.g. /content/RDD2022-1)
├── data.yaml          ← ships with the export; class names added by the rename cell
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/              (may or may not be present)
```

## Preparation checklist

- [x] Download the Roboflow YOLO export
- [x] Visual audit → name classes 0–4 → write into `data.yaml`
- [x] EDA: class counts confirm `repair` is the rare class
- [ ] (Week 3) Rebalance / augment for the `repair` class
- [ ] (Week 4) Small local Houston phone-photo test set for qualitative domain check