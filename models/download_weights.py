"""
download_weights.py — fetch the pretrained YOLO11 weights this project builds on.

Run once in your environment (Colab / Kaggle / local):
    python models/download_weights.py

Note: Ultralytics also auto-downloads a checkpoint the first time you reference it
(e.g. YOLO('yolo11s.pt')), so this script is a convenience / offline-prep step.
Weights are pulled from the official Ultralytics GitHub release assets.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

# yolo11s = primary model; yolo11n = fast smoke tests; yolo11m = accuracy upgrade path
MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
DEST = Path(__file__).parent


def main():
    for name in MODELS:
        print(f"Fetching {name} ...")
        YOLO(name)                       # triggers download into the ultralytics cache
        cached = Path(name)
        if cached.exists() and not (DEST / name).exists():
            shutil.move(str(cached), DEST / name)
    print(f"\nWeights ready in {DEST}/")
    print("Primary model for fine-tuning: yolo11s.pt")


if __name__ == "__main__":
    main()
