# Offroad Semantic Segmentation

SegFormer-B2 based semantic segmentation for 10 off-road terrain classes.

## Requirements

```
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm albumentations transformers
```

## Dataset Structure

```
Training/
  Offroad_Segmentation_Training_Dataset/
    train/
      Color_Images/
      Segmentation/
    val/
      Color_Images/
      Segmentation/

test_public_80/
  test_public_80/
    Color_Images/
    Segmentation/
```

## Training

```bash
cd Scripts
python train_segmentation.py
```

The script uses hardcoded configuration — edit `main()` in the file to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Batch size per GPU |
| `w x h` | 640 x 352 | Input resolution (must be divisible by 32) |
| `lr` | 1e-4 | Learning rate |
| `n_epochs` | 60 | Total training epochs |
| `num_workers` | 4 | DataLoader workers |

**Outputs:**
- `Scripts/segformer_b2_best.pth` — Best model checkpoint (by val mIoU)
- `Scripts/segformer_b2_domain_aware.pth` — Final epoch checkpoint
- `Scripts/train_stats/` — Loss/IoU/Dice curves and metrics log

## Testing / Inference

```bash
cd Scripts
python test_segmentation.py
```

**Command-line arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `Scripts/segformer_b2_best.pth` | Path to trained weights |
| `--data_dir` | `../test_public_80/test_public_80` | Path to test dataset |
| `--output_dir` | `./predictions` | Where to save outputs |
| `--batch_size` | 4 | Inference batch size |
| `--num_samples` | 5 | Number of side-by-side comparison images to save |

**Example with custom paths:**

```bash
python test_segmentation.py --model_path ./segformer_b2_best.pth --data_dir ../test_public_80/test_public_80 --batch_size 8 --num_samples 10
```

**Outputs:**
- `predictions/masks/` — Raw prediction masks (class IDs 0-9)
- `predictions/masks_color/` — Colored RGB prediction masks
- `predictions/comparisons/` — Side-by-side input / ground truth / prediction images
- `predictions/evaluation_metrics.txt` — Per-class IoU summary
- `predictions/per_class_metrics.png` — IoU bar chart

## Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Trees | 5 | Flowers |
| 1 | Lush Bushes | 6 | Logs |
| 2 | Dry Grass | 7 | Rocks |
| 3 | Dry Bushes | 8 | Landscape |
| 4 | Ground Clutter | 9 | Sky |
