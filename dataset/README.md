# Your dataset goes here

This folder is a placeholder — **no dataset is bundled with this
project**, and anything you put here (except this file) is git-ignored.

Expected layout, one sub-folder per sign:

```text
dataset/
├── A/        *.jpg, *.png, ...
├── B/
└── hello/
```

Then:

```bash
python scripts/check_dataset.py --dataset dataset   # validate
python train.py --dataset dataset --epochs 20       # train
```

See [../docs/dataset.md](../docs/dataset.md) for how much data you
need, where to get it, and the exact rules the loader enforces.
