"""
train.py — fine-tune YOLO11 on RDD2022 for road-damage detection.

Examples
--------
# Baseline (Week 2): short run to confirm the pipeline
python src/train.py --data configs/data.yaml --model yolo11s.pt --epochs 5

# Full run (Week 3): more epochs + stronger augmentation for the rare pothole class
python src/train.py --data configs/data.yaml --model yolo11s.pt --epochs 100 --aug

# Upgrade path if mAP is low
python src/train.py --data configs/data.yaml --model yolo11m.pt --epochs 100 --aug

The --model can be a pretrained checkpoint (yolo11s.pt, auto-downloads) or an architecture
YAML (yolo11s.yaml) to train from scratch.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO11 on RDD2022")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="checkpoint (.pt) or architecture (.yaml)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25, help="early-stopping patience")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu', or None to auto-select")
    ap.add_argument("--name", default="roadwatch")
    ap.add_argument("--aug", action="store_true", help="enable stronger augmentation for imbalance")
    args = ap.parse_args()

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, name=args.name, plots=True,
    )
    if args.aug:
        # heavier augmentation helps the rare pothole class generalize
        train_kwargs.update(
            mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            fliplr=0.5, degrees=5.0, translate=0.1, scale=0.5,
        )

    results = model.train(**train_kwargs)
    print("\nDone. Best weights:", results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
