# Models

This project fine-tunes **Ultralytics YOLO11**. Two things live here:

| File | What it is |
|------|-----------|
| `yolo11.yaml` | The YOLO11 **architecture** definition (all scales n/s/m/l/x). Ships with the code so the model is fully reproducible. |
| `download_weights.py` | Fetches the **pretrained weights** (`yolo11n/s/m.pt`) that fine-tuning starts from. |

## Which model?

- **`yolo11s.pt` — primary.** Best speed/accuracy balance that still fine-tunes on a free GPU.
- `yolo11n.pt` — fast smoke tests / debugging.
- `yolo11m.pt` — accuracy upgrade path if mAP falls short of target.

## Getting the weights

The pretrained `.pt` files are **not committed** (they're large binaries hosted on Ultralytics'
GitHub release assets). Get them either way:

**Automatic** — Ultralytics downloads the checkpoint the first time you reference it, so
`python src/train.py --model yolo11s.pt ...` just works. No manual step needed.

**Manual (optional, for offline prep):**
```bash
python models/download_weights.py     # pulls yolo11n/s/m.pt into this folder
```

> Note: this couldn't be pre-bundled from the build sandbox because that environment blocks
> GitHub's `release-assets.githubusercontent.com` host. In Colab, Kaggle, or a normal machine
> the download works with no setup.

## Training from scratch (no download)

You can also build the model straight from the architecture YAML — this needs no weights at all
and is how the pipeline was smoke-tested:
```bash
python src/train.py --model yolo11n.yaml ...   # random init, trains from scratch
```
Transfer learning from `yolo11s.pt` is strongly preferred for real accuracy.
