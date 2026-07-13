"""
detect.py — run a trained RoadWatch model on images or video.

Saves annotated output (boxes drawn) and writes a per-image damage-count CSV report,
the artifact a public-works team would actually use.

Examples
--------
python src/detect.py --weights runs/detect/roadwatch/weights/best.pt --source demo_images/
python src/detect.py --weights best.pt --source houston_drive.mp4 --conf 0.35
"""
import argparse
import collections
import csv
import os
from ultralytics import YOLO

# Class names come from the trained model at runtime (supports 4- or 5-class sets).
# Rough severity weighting for a simple per-image score; unknown classes default to 1.
SEVERITY = {"longitudinal_crack": 1, "transverse_crack": 1, "alligator_crack": 2,
            "pothole": 3, "repair": 1}


def main():
    ap = argparse.ArgumentParser(description="Detect road damage and write a report")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="report", help="run name under runs/detect/")
    ap.add_argument("--csv", default="damage_report.csv")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names                      # class names baked into the checkpoint
    CLASSES = [names[i] for i in sorted(names)]

    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                            save=True, name=args.out)

    rows = []
    for r in results:
        counts = collections.Counter(int(c) for c in r.boxes.cls.tolist())
        per_class = {CLASSES[i]: counts.get(i, 0) for i in range(len(CLASSES))}
        score = sum(per_class[c] * SEVERITY.get(c, 1) for c in CLASSES)
        rows.append({"image": os.path.basename(r.path), **per_class,
                     "total_defects": sum(per_class.values()), "severity_score": score})

    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")
        print("Annotated images saved under runs/detect/" + args.out)
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
