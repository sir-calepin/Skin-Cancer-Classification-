# Models

This folder is for locally generated model checkpoints and final model weights.

## Folder roles

```text
models/
├── checkpoints/  # Best checkpoints from intermediate experiments
└── final/        # Selected final model checkpoint
```

## Expected final model

After running:

```bash
python scripts/train_resnet50_final.py
```

the project saves:

```text
models/final/resnet50_final.pth
```

## Why model files are excluded

Model files can be large, and they are generated artifacts rather than source code. The root `.gitignore` excludes `.pth`, `.pt`, `.ckpt`, and similar formats.

If you want to publish a checkpoint, use a release asset, Git LFS, a model registry, or a cloud-storage link rather than a normal Git commit.
