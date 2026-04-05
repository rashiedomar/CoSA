#!/usr/bin/env python3
"""
Create "Hero" Comparison Visualizations: Baseline vs CoSA

Finds and visualizes the best examples where CoSA fixes baseline errors:
1. Baseline missed a building (FN) but CoSA found it
2. Baseline hallucinated a change (FP) but CoSA removed it

Layout: 5 Columns
- Col 1: Image T1 (Before)
- Col 2: Image T2 (After)
- Col 3: Ground Truth (GT)
- Col 4: Baseline Prediction (Red = Wrong, White = Right)
- Col 5: CoSA Prediction (Green box highlighting where it fixed the error)
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from scipy.ndimage import label, find_objects

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from change_detection.models.siamese_unet import SiameseUNet, SiameseUNetCoSA
from change_detection.scripts.data.levir_dataset_fixed import LEVIRCDDatasetFixed


def load_model(checkpoint_path, variant, device):
    """Load model from checkpoint."""
    if variant == 'baseline':
        model = SiameseUNet(in_channels=3, n_classes=1, base_channels=64, fusion='diff')
    elif variant == 'cosa_v3':
        model = SiameseUNetCoSA(
            in_channels=3, n_classes=1, base_channels=64, fusion='diff',
            topk=32, use_multiscale=True, use_learnable_gate=True
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def predict(model, img1, img2, device, variant='baseline'):
    """Generate prediction from model."""
    with torch.no_grad():
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
        else:
            img1 = img1.to(device)
            img2 = img2.to(device)
        
        if variant in ['cosa', 'cosa_v3']:
            output, _ = model(img1, img2)
        else:
            output = model(img1, img2)
        
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        
        return pred, pred_binary


def denormalize_image(img_tensor):
    """Denormalize ImageNet-normalized image for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if img_tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    if img.dim() == 4:
        img = img[0]
    return img.permute(1, 2, 0).cpu().numpy()


def compute_error_regions(pred, gt):
    """Compute error regions: FP, FN, TP, TN."""
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    
    # False Positives: pred=1, gt=0
    fp = ((pred_bin == 1) & (gt_bin == 0)).astype(np.float32)
    # False Negatives: pred=0, gt=1
    fn = ((pred_bin == 0) & (gt_bin == 1)).astype(np.float32)
    # True Positives: pred=1, gt=1
    tp = ((pred_bin == 1) & (gt_bin == 1)).astype(np.float32)
    # True Negatives: pred=0, gt=0
    tn = ((pred_bin == 0) & (gt_bin == 0)).astype(np.float32)
    
    return fp, fn, tp, tn


def find_fixed_regions(pred_baseline, pred_cosa, gt):
    """Find regions where CoSA fixed baseline errors."""
    fp_b, fn_b, tp_b, tn_b = compute_error_regions(pred_baseline, gt)
    fp_c, fn_c, tp_c, tn_c = compute_error_regions(pred_cosa, gt)
    
    # CoSA fixed False Negatives: baseline missed, CoSA found
    fixed_fn = (fn_b > 0.5) & (tp_c > 0.5)
    
    # CoSA fixed False Positives: baseline hallucinated, CoSA removed
    fixed_fp = (fp_b > 0.5) & (tn_c > 0.5)
    
    return fixed_fn, fixed_fp


def get_bounding_boxes(mask, min_area=100):
    """Get bounding boxes for connected components in mask."""
    labeled_mask, num_features = label(mask.astype(bool))
    objects = find_objects(labeled_mask)
    boxes = []
    for i, obj in enumerate(objects):
        if obj is not None:
            area = (labeled_mask[obj] == i+1).sum()
            if area >= min_area:
                minr, minc = obj[0].start, obj[1].start
                maxr, maxc = obj[0].stop, obj[1].stop
                boxes.append((minc, minr, maxc, maxr))  # (x1, y1, x2, y2)
    return boxes


def create_hero_visualization(img1, img2, gt, pred_baseline, pred_cosa, name, output_path, 
                               baseline_f1=None, cosa_f1=None, improvement=None):
    """Create Hero comparison visualization with 5 columns."""
    # Denormalize images
    img1_np = denormalize_image(img1)
    img2_np = denormalize_image(img2)
    
    # Convert GT to numpy if tensor
    if isinstance(gt, torch.Tensor):
        gt_np = gt.squeeze().cpu().numpy()
    else:
        gt_np = gt
    
    # Ensure binary masks
    gt_bin = (gt_np > 0.5).astype(np.float32)
    pred_baseline_bin = (pred_baseline > 0.5).astype(np.float32)
    pred_cosa_bin = (pred_cosa > 0.5).astype(np.float32)
    
    # Compute error regions
    fp_b, fn_b, tp_b, tn_b = compute_error_regions(pred_baseline_bin, gt_bin)
    fp_c, fn_c, tp_c, tn_c = compute_error_regions(pred_cosa_bin, gt_bin)
    
    # Find where CoSA fixed errors
    fixed_fn, fixed_fp = find_fixed_regions(pred_baseline_bin, pred_cosa_bin, gt_bin)
    
    # Get bounding boxes for fixed regions
    fixed_fn_boxes = get_bounding_boxes(fixed_fn, min_area=50)
    fixed_fp_boxes = get_bounding_boxes(fixed_fp, min_area=50)
    
    # Create figure with 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Col 1: T1 Image
    axes[0].imshow(img1_np)
    axes[0].set_title('T1 (Before)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Col 2: T2 Image
    axes[1].imshow(img2_np)
    axes[1].set_title('T2 (After)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Col 3: Ground Truth
    axes[2].imshow(img1_np)
    axes[2].imshow(gt_bin, alpha=0.6, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Col 4: Baseline Prediction (Red = Wrong, White = Right)
    # Create RGB visualization: White background, Green for TP, Red for FP/FN
    baseline_rgb = np.ones((*pred_baseline_bin.shape, 3))
    
    # True Positives: Green
    baseline_rgb[:, :, 0] = np.where(tp_b > 0.5, 0.0, baseline_rgb[:, :, 0])
    baseline_rgb[:, :, 1] = np.where(tp_b > 0.5, 1.0, baseline_rgb[:, :, 1])
    baseline_rgb[:, :, 2] = np.where(tp_b > 0.5, 0.0, baseline_rgb[:, :, 2])
    
    # False Positives: Red
    baseline_rgb[:, :, 0] = np.where(fp_b > 0.5, 1.0, baseline_rgb[:, :, 0])
    baseline_rgb[:, :, 1] = np.where(fp_b > 0.5, 0.0, baseline_rgb[:, :, 1])
    baseline_rgb[:, :, 2] = np.where(fp_b > 0.5, 0.0, baseline_rgb[:, :, 2])
    
    # False Negatives: Red (darker)
    baseline_rgb[:, :, 0] = np.where(fn_b > 0.5, 0.8, baseline_rgb[:, :, 0])
    baseline_rgb[:, :, 1] = np.where(fn_b > 0.5, 0.2, baseline_rgb[:, :, 1])
    baseline_rgb[:, :, 2] = np.where(fn_b > 0.5, 0.2, baseline_rgb[:, :, 2])
    
    # Overlay on image
    axes[3].imshow(img1_np)
    axes[3].imshow(baseline_rgb, alpha=0.7)
    title = 'Baseline Prediction'
    if baseline_f1 is not None:
        title += f'\nF1: {baseline_f1:.3f}'
    axes[3].set_title(title, fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    # Add legend for baseline
    green_patch = mpatches.Patch(color='green', label='Correct (TP)')
    red_patch = mpatches.Patch(color='red', label='Wrong (FP/FN)')
    axes[3].legend(handles=[green_patch, red_patch], loc='upper right', fontsize=8)
    
    # Col 5: CoSA Prediction with Green boxes highlighting fixes
    # Create RGB visualization: White background, Green for TP, Red for FP/FN
    cosa_rgb = np.ones((*pred_cosa_bin.shape, 3))
    
    # True Positives: Green
    cosa_rgb[:, :, 0] = np.where(tp_c > 0.5, 0.0, cosa_rgb[:, :, 0])
    cosa_rgb[:, :, 1] = np.where(tp_c > 0.5, 1.0, cosa_rgb[:, :, 1])
    cosa_rgb[:, :, 2] = np.where(tp_c > 0.5, 0.0, cosa_rgb[:, :, 2])
    
    # False Positives: Red
    cosa_rgb[:, :, 0] = np.where(fp_c > 0.5, 1.0, cosa_rgb[:, :, 0])
    cosa_rgb[:, :, 1] = np.where(fp_c > 0.5, 0.0, cosa_rgb[:, :, 1])
    cosa_rgb[:, :, 2] = np.where(fp_c > 0.5, 0.0, cosa_rgb[:, :, 2])
    
    # False Negatives: Red (darker)
    cosa_rgb[:, :, 0] = np.where(fn_c > 0.5, 0.8, cosa_rgb[:, :, 0])
    cosa_rgb[:, :, 1] = np.where(fn_c > 0.5, 0.2, cosa_rgb[:, :, 1])
    cosa_rgb[:, :, 2] = np.where(fn_c > 0.5, 0.2, cosa_rgb[:, :, 2])
    
    # Overlay on image
    axes[4].imshow(img1_np)
    axes[4].imshow(cosa_rgb, alpha=0.7)
    
    # Draw green boxes around fixed regions
    for x1, y1, x2, y2 in fixed_fn_boxes:
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='lime', facecolor='none')
        axes[4].add_patch(rect)
    
    for x1, y1, x2, y2 in fixed_fp_boxes:
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='cyan', facecolor='none')
        axes[4].add_patch(rect)
    
    title = 'CoSA Prediction'
    if cosa_f1 is not None:
        title += f'\nF1: {cosa_f1:.3f}'
    if improvement is not None:
        title += f' (+{improvement:.3f})'
    axes[4].set_title(title, fontsize=14, fontweight='bold')
    axes[4].axis('off')
    
    # Add legend for CoSA
    green_patch = mpatches.Patch(color='green', label='Correct (TP)')
    red_patch = mpatches.Patch(color='red', label='Wrong (FP/FN)')
    lime_patch = mpatches.Patch(color='lime', label='Fixed FN', fill=False)
    cyan_patch = mpatches.Patch(color='cyan', label='Fixed FP', fill=False)
    axes[4].legend(handles=[green_patch, red_patch, lime_patch, cyan_patch], 
                   loc='upper right', fontsize=8)
    
    # Add overall title
    title_str = f'Sample: {name}'
    if baseline_f1 is not None and cosa_f1 is not None:
        title_str += f' | Baseline F1: {baseline_f1:.3f} → CoSA F1: {cosa_f1:.3f}'
    fig.suptitle(title_str, fontsize=16, fontweight='bold', y=1.02)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return len(fixed_fn_boxes), len(fixed_fp_boxes)


def find_best_samples(dataset_dir, baseline_model, cosa_model, device, 
                       target_fn_samples=2, target_fp_samples=2):
    """Find best samples where CoSA fixes FN and FP."""
    dataset = LEVIRCDDatasetFixed(
        root_dir=dataset_dir,
        split='test',
        base_size=512,
        augment=False,
        eval_full_res=True
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    fn_candidates = []  # Samples where CoSA fixed FN
    fp_candidates = []  # Samples where CoSA fixed FP
    
    print(f"\nScanning {len(dataset)} samples...")
    
    for idx, (img1, img2, label, name) in enumerate(dataloader):
        if isinstance(label, torch.Tensor):
            label_np = label.squeeze().cpu().numpy()
        else:
            label_np = label
        label_bin = (label_np > 0.5).astype(np.float32)
        
        # Skip no-change images
        if label_bin.sum() == 0:
            continue
        
        # Generate predictions
        _, pred_baseline = predict(baseline_model, img1, img2, device, 'baseline')
        _, pred_cosa = predict(cosa_model, img1, img2, device, 'cosa_v3')
        
        pred_baseline_bin = (pred_baseline > 0.5).astype(np.float32)
        pred_cosa_bin = (pred_cosa > 0.5).astype(np.float32)
        
        # Compute metrics
        fp_b, fn_b, tp_b, tn_b = compute_error_regions(pred_baseline_bin, label_bin)
        fp_c, fn_c, tp_c, tn_c = compute_error_regions(pred_cosa_bin, label_bin)
        
        # Find fixed regions
        fixed_fn, fixed_fp = find_fixed_regions(pred_baseline_bin, pred_cosa_bin, label_bin)
        
        fn_fixed_count = fixed_fn.sum()
        fp_fixed_count = fixed_fp.sum()
        
        # Compute F1 scores
        baseline_f1 = 2 * tp_b.sum() / (tp_b.sum() + fp_b.sum() + fn_b.sum() + 1e-8)
        cosa_f1 = 2 * tp_c.sum() / (tp_c.sum() + fp_c.sum() + fn_c.sum() + 1e-8)
        improvement = cosa_f1 - baseline_f1
        
        # Handle name
        if isinstance(name, (tuple, list)):
            name_str = name[0] if len(name) > 0 else f"sample_{idx}"
        else:
            name_str = str(name)
        name_str = name_str.replace("'", "").replace("(", "").replace(")", "").replace(",", "")
        
        # Collect FN fixes (CoSA found what baseline missed)
        if fn_fixed_count > 100:  # Minimum area threshold
            fn_candidates.append({
                'idx': idx,
                'name': name_str,
                'img1': img1,
                'img2': img2,
                'label': label,
                'pred_baseline': pred_baseline,
                'pred_cosa': pred_cosa,
                'baseline_f1': baseline_f1,
                'cosa_f1': cosa_f1,
                'improvement': improvement,
                'fn_fixed': fn_fixed_count,
                'fp_fixed': fp_fixed_count,
            })
        
        # Collect FP fixes (CoSA removed what baseline hallucinated)
        if fp_fixed_count > 100:  # Minimum area threshold
            fp_candidates.append({
                'idx': idx,
                'name': name_str,
                'img1': img1,
                'img2': img2,
                'label': label,
                'pred_baseline': pred_baseline,
                'pred_cosa': pred_cosa,
                'baseline_f1': baseline_f1,
                'cosa_f1': cosa_f1,
                'improvement': improvement,
                'fn_fixed': fn_fixed_count,
                'fp_fixed': fp_fixed_count,
            })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(dataset)}")
    
    # Sort and select best
    fn_candidates.sort(key=lambda x: x['fn_fixed'], reverse=True)
    fp_candidates.sort(key=lambda x: x['fp_fixed'], reverse=True)
    
    print(f"\n✅ Found {len(fn_candidates)} FN-fix candidates, {len(fp_candidates)} FP-fix candidates")
    
    return fn_candidates[:target_fn_samples], fp_candidates[:target_fp_samples]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Hero Comparison Visualizations')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to LEVIR-CD dataset root')
    parser.add_argument('--baseline_checkpoint', type=str,
                       default='results/baseline_bs8/checkpoint_best.pth',
                       help='Path to baseline checkpoint')
    parser.add_argument('--cosa_checkpoint', type=str,
                       default='results/cosa_v3_residual_multiscale/checkpoint_best.pth',
                       help='Path to CoSA checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='results/baseline_cosa_visualizations/hero_comparisons',
                       help='Output directory for visualizations')
    parser.add_argument('--num_fn_samples', type=int, default=2,
                       help='Number of FN-fix samples to visualize')
    parser.add_argument('--num_fp_samples', type=int, default=2,
                       help='Number of FP-fix samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hero Comparison Visualization: Baseline vs CoSA")
    print("=" * 70)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Baseline: {args.baseline_checkpoint}")
    print(f"CoSA: {args.cosa_checkpoint}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    baseline_model = load_model(args.baseline_checkpoint, 'baseline', args.device)
    cosa_model = load_model(args.cosa_checkpoint, 'cosa_v3', args.device)
    print("✅ Models loaded")
    
    # Find best samples
    fn_samples, fp_samples = find_best_samples(
        args.dataset_dir, baseline_model, cosa_model, args.device,
        target_fn_samples=args.num_fn_samples,
        target_fp_samples=args.num_fp_samples
    )
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # FN fixes (CoSA found what baseline missed)
    for i, sample in enumerate(fn_samples):
        output_path = output_dir / f"hero_fn_fix_{i+1:02d}_{sample['name']}.png"
        fn_boxes, fp_boxes = create_hero_visualization(
            sample['img1'], sample['img2'], sample['label'],
            sample['pred_baseline'], sample['pred_cosa'],
            sample['name'], output_path,
            baseline_f1=sample['baseline_f1'],
            cosa_f1=sample['cosa_f1'],
            improvement=sample['improvement']
        )
        print(f"  FN Fix #{i+1}: {sample['name']} | Fixed {fn_boxes} FN regions, {fp_boxes} FP regions")
    
    # FP fixes (CoSA removed what baseline hallucinated)
    for i, sample in enumerate(fp_samples):
        output_path = output_dir / f"hero_fp_fix_{i+1:02d}_{sample['name']}.png"
        fn_boxes, fp_boxes = create_hero_visualization(
            sample['img1'], sample['img2'], sample['label'],
            sample['pred_baseline'], sample['pred_cosa'],
            sample['name'], output_path,
            baseline_f1=sample['baseline_f1'],
            cosa_f1=sample['cosa_f1'],
            improvement=sample['improvement']
        )
        print(f"  FP Fix #{i+1}: {sample['name']} | Fixed {fn_boxes} FN regions, {fp_boxes} FP regions")
    
    print(f"\n✅ Generated {len(fn_samples) + len(fp_samples)} hero comparisons in {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
