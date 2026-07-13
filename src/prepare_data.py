"""
prepare_data.py — turn raw RDD2022 (VOC XML) into a YOLO-ready dataset.

Steps:
  1. Convert VOC XML -> YOLO txt (only the 4 target classes)
  2. Copy images + labels into an 80/20 train/val split
  3. Drop images that end up with no annotations
  4. Write configs/data.yaml

Example (per downloaded country subset with VOC structure images/ + annotations/xmls/):
  python src/prepare_data.py --country_dirs data/raw/United_States data/raw/Japan \
                             --out data --val_frac 0.2

If you downloaded a pre-converted YOLO mirror (Roboflow/Kaggle), you can skip this script.
"""
import argparse
import random
import shutil
from pathlib import Path

from voc_to_yolo import convert_one, CLASS_MAP  # local import


def gather(country_dirs):
    """Yield (image_path, xml_path) pairs from RDD-style country folders."""
    pairs = []
    for cd in country_dirs:
        cd = Path(cd)
        img_dir = cd / "images"
        xml_dir = cd / "annotations" / "xmls"
        if not img_dir.is_dir() or not xml_dir.is_dir():
            print(f"  ! skipping {cd} (expected images/ and annotations/xmls/)")
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            xml = xml_dir / (img.stem + ".xml")
            if xml.exists():
                pairs.append((img, xml))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs = gather(args.country_dirs)
    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    kept = {"train": 0, "val": 0}
    for split, items in splits.items():
        tmp_lbl = out / "labels" / split
        for img, xml in items:
            n_boxes = convert_one(xml, tmp_lbl)
            if n_boxes == 0:                          # drop empty images
                (tmp_lbl / (xml.stem + ".txt")).unlink(missing_ok=True)
                continue
            shutil.copy(img, out / "images" / split / img.name)
            kept[split] += 1

    # write data.yaml
    names = {v: k for k, v in CLASS_MAP.items()}
    label_names = {0: "longitudinal_crack", 1: "transverse_crack",
                   2: "alligator_crack", 3: "pothole"}
    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {label_names[i]}\n" for i in range(4))
    )
    Path("configs").mkdir(exist_ok=True)
    Path("configs/data.yaml").write_text(yaml_text)

    print(f"Prepared dataset -> {out}")
    print(f"  train: {kept['train']} images   val: {kept['val']} images")
    print("  wrote configs/data.yaml")


if __name__ == "__main__":
    main()
