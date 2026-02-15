"""
Segmentation Test/Inference Script
Evaluates a trained SegFormer-B2 model on test data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Mask Conversion (matches train_segmentation_improved.py)
# ============================================================================

# Strategic Class Remapping (10 Competition Classes)
value_map = {
    0: 8,        # background -> Landscape
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    600: 5,      # Flowers
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = 10
id2label = {
    0: "Trees", 1: "Lush Bushes", 2: "Dry Grass", 3: "Dry Bushes",
    4: "Ground Clutter", 5: "Flowers", 6: "Logs", 7: "Rocks",
    8: "Landscape", 9: "Sky"
}
label2id = {v: k for k, v in id2label.items()}

# Class names for visualization (ordered by class ID)
class_names = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [255, 105, 180],  # Flowers - hot pink
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset (albumentations-based, matching training)
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = Image.open(mask_path)
        mask = np.array(convert_mask(mask))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask, data_id


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    fig, ax = plt.subplots(figsize=(10, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Test Function
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation prediction/inference script (SegFormer-B2)')
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'segformer_b2_best.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default=os.path.join(script_dir, '..', 'test_public_80', 'test_public_80'),
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions (must match training)
    w = 448
    h = 448

    # Transforms (matching training val_transform)
    test_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    testset = MaskDataset(data_dir=args.data_dir, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Loaded {len(testset)} samples")

    # Load SegFormer-B2 model
    print(f"Loading SegFormer-B2 from {args.model_path}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation
    print(f"\nRunning evaluation and saving predictions for all {len(testset)} images...")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass through SegFormer
            outputs = model(imgs).logits
            outputs = F.interpolate(outputs, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_long = labels.long()
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics
            iou, class_iou = compute_iou(outputs, labels_long, num_classes=n_classes)
            dice, class_dice = compute_dice(outputs, labels_long, num_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels_long)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-9)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_long[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # Aggregate results
    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)

    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        'mean_iou': mean_iou,
        'class_iou': avg_class_iou
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Pixel Accuracy:    {mean_pixel_acc:.4f}")
    print("=" * 50)
    print("\nPer-Class IoU:")
    for i, name in enumerate(class_names):
        iou_val = avg_class_iou[i]
        print(f"  {name:<20}: {iou_val:.4f}" if not np.isnan(iou_val) else f"  {name:<20}: N/A")
    print("=" * 50)

    # Save all results
    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(testset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()
