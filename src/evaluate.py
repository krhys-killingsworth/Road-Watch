"""
evaluate.py — report detection metrics on the validation split.

Prints overall mAP plus per-class AP, precision, and recall — not a single inflated number.
Targets (from the proposal): mAP@0.5 >= 0.55 and pothole recall >= 0.60.

Example
-------
python src/evaluate.py --weights runs/detect/roadwatch/weights/best.pt --data configs/data.yaml
"""
import argparse
from ultralytics import YOLO

# Class names are read from the dataset yaml at runtime (supports 4- or 5-class sets)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained RoadWatch model")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import yaml as _yaml
    names = _yaml.safe_load(open(args.data))["names"]
    CLASSES = list(names.values()) if isinstance(names, dict) else list(names)

    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    print("\n================ RoadWatch metrics ================")
    print(f"mAP@0.5       : {m.box.map50:.4f}   (target >= 0.55)")
    print(f"mAP@0.5:0.95  : {m.box.map:.4f}")
    print(f"precision     : {m.box.mp:.4f}")
    print(f"recall        : {m.box.mr:.4f}")
    print("\nPer-class AP@0.5:")
    # m.box.maps is per-class mAP@0.5:0.95; ap50 gives per-class AP@0.5 when available
    try:
        ap50 = m.box.ap50
        for i, name in enumerate(CLASSES):
            print(f"  {name:20s}: {ap50[i]:.4f}")
    except Exception:
        for i, name in enumerate(CLASSES):
            print(f"  {name:20s}: {m.box.maps[i]:.4f}")
    print("===================================================")


if __name__ == "__main__":
    main()
