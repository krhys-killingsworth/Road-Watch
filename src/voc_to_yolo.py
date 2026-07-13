"""
voc_to_yolo.py — convert RDD2022 Pascal VOC XML annotations to YOLO txt format.

RDD2022 ships bounding boxes as VOC XML; Ultralytics YOLO expects one .txt per image
with lines of:  class_id  x_center  y_center  width  height   (all normalized 0-1).

Usage
-----
    python voc_to_yolo.py --xml_dir path/to/annotations --img_dir path/to/images \
                          --out_dir path/to/labels

Only the 4 target classes (D00, D10, D20, D40) are kept; other tags are skipped.
"""
import argparse
import os
from pathlib import Path
from xml.etree import ElementTree as ET

# RDD2022 damage code -> YOLO class id
CLASS_MAP = {
    "D00": 0,  # longitudinal crack
    "D10": 1,  # transverse crack
    "D20": 2,  # alligator crack
    "D40": 3,  # pothole
}


def convert_one(xml_path: Path, out_dir: Path) -> int:
    """Convert a single VOC XML file. Returns the number of boxes written."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    if img_w == 0 or img_h == 0:
        return 0

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue  # skip D43/D44/D50 etc.
        cls = CLASS_MAP[name]
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)

        # to normalized center-x, center-y, w, h
        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    out_path = out_dir / (xml_path.stem + ".txt")
    out_path.write_text("\n".join(lines))
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="folder of VOC .xml files")
    ap.add_argument("--out_dir", required=True, help="output folder for YOLO .txt files")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    total_boxes, total_files = 0, 0
    for xf in xml_files:
        n = convert_one(xf, out_dir)
        total_boxes += n
        total_files += 1

    print(f"Converted {total_files} files -> {out_dir}  ({total_boxes} boxes kept)")


if __name__ == "__main__":
    main()
