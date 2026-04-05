#!/usr/bin/env python3
"""
Fine-tune STANet with CoSA (Correlation-guided Spatial Attention) block
Only the CoSA block is trainable, STANet backbone is frozen
"""

import sys
import os
from pathlib import Path
import argparse

# Add current directory to path
STANET_DIR = Path(__file__).parent
sys.path.insert(0, str(STANET_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import logging
import copy

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.cosa_block import CoSABlock
from util.metrics import RunningMetrics

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = RESEARCH_ROOT.parent


def setup_logging(log_dir):
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'finetune_cosa.log')
    
    logger = logging.getLogger('STANet_CoSA')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class STANetWithCoSA(nn.Module):
    """
    Wrapper that adds CoSA block to frozen STANet
    """
    
    def __init__(self, stanet_model, cosa_scales=[8, 16]):
        super(STANetWithCoSA, self).__init__()
        
        # Store STANet model
        self.stanet = stanet_model
        
        # Freeze STANet networks completely
        for param in self.stanet.netF.parameters():
            param.requires_grad = False
        for param in self.stanet.netA.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.stanet.netF.eval()
        self.stanet.netA.eval()
        
        # Add CoSA block (trainable)
        self.cosa = CoSABlock(in_channels=64, scales=cosa_scales)
        
        # Final prediction layer (trainable)
        self.pred_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, A, B):
        """
        Args:
            A: [B, 3, H, W] image at time T1
            B: [B, 3, H, W] image at time T2
        Returns:
            pred: [B, 1, H, W] change prediction
            enhanced_diff: [B, 64, H', W'] enhanced difference features
        """
        with torch.no_grad():
            # Extract features using frozen STANet
            # netF outputs [B, 64, H', W'] where H', W' depends on the decoder
            feat_A = self.stanet.netF(A)  # [B, 64, H', W']
            feat_B = self.stanet.netF(B)
            
            # Apply attention module (frozen)
            feat_A, feat_B = self.stanet.netA(feat_A, feat_B)
            # After attention, still [B, 64, H', W']
        
        # CoSA block (trainable) - processes at multiple scales
        enhanced_diff = self.cosa(feat_A, feat_B)  # [B, 64, H', W']
        
        # Upsample to input resolution if needed
        if enhanced_diff.shape[2:] != A.shape[2:]:
            enhanced_diff = F.interpolate(enhanced_diff, size=A.shape[2:], 
                                         mode='bilinear', align_corners=False)
        
        # Final prediction
        pred = self.pred_conv(enhanced_diff)  # [B, 1, H, W]
        
        return pred, enhanced_diff


def validate(model, val_dataset, device, logger, epoch):
    """Validate model and return metrics"""
    model.eval()
    metrics = RunningMetrics(2)
    
    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            A = data['A'].to(device)
            B = data['B'].to(device)
            L = data['L'].to(device).long()
            
            pred, _ = model(A, B)
            
            # Convert to binary prediction
            pred_binary = (torch.sigmoid(pred) > 0.5).long()  # [B, 1, H, W]
            
            # Handle shapes for metrics
            if L.ndim == 4 and L.shape[1] == 1:
                L = L.squeeze(1)  # [B, H, W]
            if pred_binary.ndim == 4 and pred_binary.shape[1] == 1:
                pred_binary = pred_binary.squeeze(1)  # [B, H, W]
            
            # Update metrics - process each sample in batch
            L_np = L.detach().cpu().numpy()
            pred_np = pred_binary.detach().cpu().numpy()
            
            # Process batch by batch
            for b in range(L_np.shape[0]):
                metrics.update(np.array([L_np[b]]), np.array([pred_np[b]]))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Validating batch {i+1}/{len(val_dataset)}")
    
    scores = metrics.get_scores()
    return scores


def train_epoch(model, train_dataset, criterion, optimizer, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, data in enumerate(train_dataset):
        A = data['A'].to(device)
        B = data['B'].to(device)
        L = data['L'].to(device).float()
        
        # Forward pass
        pred, _ = model(A, B)
        
        # Prepare labels for loss
        if L.ndim == 3:
            L = L.unsqueeze(1)  # [B, 1, H, W]
        
        # Resize label if needed
        if L.shape[2:] != pred.shape[2:]:
            L = F.interpolate(L, size=pred.shape[2:], mode='nearest')
        
        # Compute loss
        loss = criterion(pred, L)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {i+1}/{len(train_dataset)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Fine-tune STANet with CoSA')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='',
        help='Path to LEVIR-CD_combined_256 root containing train/ val/ test/ folders',
    )
    parser.add_argument(
        '--stanet_checkpoint_dir',
        type=str,
        default='',
        help='Path to STANet checkpoint dir (contains *_net_F.pth and *_net_A.pth)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(SCRIPT_DIR),
        help='Directory for finetuning logs and best checkpoint',
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint_epoch', type=str, default='90_F1_1_0.80165')
    parser.add_argument('--fallback_epoch', type=str, default='latest')
    args = parser.parse_args()

    dataset_candidates = [
        WORKSPACE_ROOT / 'datasets' / 'to_check_dataset' / 'LEVIR-CD_combined_256',
        RESEARCH_ROOT / 'data' / 'LEVIR-CD_combined_256',
    ]
    checkpoint_candidates = [
        WORKSPACE_ROOT / 'checkpoints_official' / 'STANet_LEVIR' / 'STANet-master'
        / 'checkpoints' / 'stanet_official_paper',
    ]

    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if dataset_root is None:
        dataset_root = next((p for p in dataset_candidates if p.exists()), None)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            'dataset_root not found. Pass --dataset_root or create one of: '
            + ', '.join(str(p) for p in dataset_candidates)
        )

    stanet_checkpoint_dir = (
        Path(args.stanet_checkpoint_dir).expanduser() if args.stanet_checkpoint_dir else None
    )
    if stanet_checkpoint_dir is None:
        stanet_checkpoint_dir = next((p for p in checkpoint_candidates if p.exists()), None)
    if stanet_checkpoint_dir is None or not stanet_checkpoint_dir.exists():
        raise FileNotFoundError(
            'stanet_checkpoint_dir not found. Pass --stanet_checkpoint_dir or create: '
            + ', '.join(str(p) for p in checkpoint_candidates)
        )

    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'
    test_dir = dataset_root / 'test'
    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            raise FileNotFoundError(f'Missing dataset split directory: {split_dir}')

    # Setup
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create output directory
    output_dir = str(Path(args.output_dir).expanduser())
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("STANet + CoSA Fine-tuning")
    logger.info("="*60)
    
    # Load STANet checkpoint
    stanet_checkpoint_dir = str(stanet_checkpoint_dir)
    
    # Find best checkpoint (epoch 90)
    best_checkpoint = os.path.join(stanet_checkpoint_dir, args.checkpoint_epoch)
    if not os.path.exists(best_checkpoint + '_net_F.pth'):
        # Try latest
        best_checkpoint = os.path.join(stanet_checkpoint_dir, args.fallback_epoch)
        logger.info(f"Best checkpoint not found, using latest: {best_checkpoint}")
    
    logger.info(f"Loading STANet from: {best_checkpoint}")
    
    # Create STANet model and load weights
    argv_save = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        opt_stanet = TrainOptions().parse()
    finally:
        sys.argv = argv_save
    opt_stanet.model = 'CDFA'
    opt_stanet.arch = 'mynet3'
    opt_stanet.SA_mode = 'BAM'
    opt_stanet.f_c = 64
    opt_stanet.ds = 1
    opt_stanet.isTrain = False
    opt_stanet.phase = 'test'
    opt_stanet.gpu_ids = str(args.gpu_id) if torch.cuda.is_available() else ''
    opt_stanet.checkpoints_dir = str(Path(stanet_checkpoint_dir).parent)
    opt_stanet.name = Path(stanet_checkpoint_dir).name
    
    stanet_model = create_model(opt_stanet)
    
    # Set checkpoint directory before setup
    stanet_model.save_dir = stanet_checkpoint_dir
    
    # Setup without loading (we'll load manually)
    opt_stanet.continue_train = False
    opt_stanet.epoch = args.checkpoint_epoch
    stanet_model.setup(opt_stanet)
    
    # Load checkpoint manually
    try:
        stanet_model.load_networks(args.checkpoint_epoch)
        logger.info(f"Loaded STANet checkpoint: {args.checkpoint_epoch}")
    except Exception as e1:
        logger.warning(f"Failed to load {args.checkpoint_epoch}: {e1}")
        try:
            stanet_model.load_networks(args.fallback_epoch)
            logger.info(f"Loaded STANet checkpoint: {args.fallback_epoch}")
        except Exception as e2:
            logger.error(f"Failed to load checkpoint: {e2}")
            import traceback
            logger.error(traceback.format_exc())
            return
    
    # Create STANet + CoSA model
    model = STANetWithCoSA(stanet_model, cosa_scales=[8, 16]).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    stanet_params = sum(p.numel() for p in model.stanet.netF.parameters()) + sum(p.numel() for p in model.stanet.netA.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters (CoSA + pred_conv): {trainable_params:,}")
    logger.info(f"Frozen STANet parameters: {stanet_params:,}")
    logger.info(f"CoSA block parameters: {sum(p.numel() for p in model.cosa.parameters()):,}")
    logger.info(f"Prediction conv parameters: {sum(p.numel() for p in model.pred_conv.parameters()):,}")
    
    # Create datasets
    train_opt = copy.deepcopy(opt_stanet)
    train_opt.dataroot = str(train_dir)
    train_opt.dataset_mode = 'changedetection'
    train_opt.batch_size = args.batch_size
    train_opt.num_threads = args.num_workers
    train_opt.serial_batches = False
    train_opt.preprocess = 'resize_and_crop'
    train_opt.load_size = 256
    train_opt.crop_size = 256
    train_opt.angle = 5
    train_opt.no_flip = False
    train_opt.phase = 'train'
    train_opt.isTrain = True
    
    val_opt = copy.deepcopy(opt_stanet)
    val_opt.dataroot = str(val_dir)
    val_opt.dataset_mode = 'changedetection'
    val_opt.batch_size = args.batch_size
    val_opt.num_threads = 0
    val_opt.serial_batches = True
    val_opt.preprocess = ''
    val_opt.angle = 0
    val_opt.no_flip = True
    val_opt.phase = 'val'
    val_opt.isTrain = False
    
    train_dataset = create_dataset(train_opt)
    val_dataset = create_dataset(val_opt)
    
    # STANet dataset already returns batches, so we use it directly
    # train_loader and val_loader are the datasets themselves
    train_loader = train_dataset
    val_loader = val_dataset
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"STANet checkpoint dir: {stanet_checkpoint_dir}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    num_epochs = args.epochs
    best_f1 = 0.0
    best_epoch = 0
    
    logger.info("="*60)
    logger.info("Starting fine-tuning...")
    logger.info(
        f"Epochs: {num_epochs}, Batch size: {args.batch_size}, "
        f"LR: {args.lr}, Weight decay: {args.weight_decay}"
    )
    logger.info("="*60)
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Validate
        logger.info(f"Validating epoch {epoch}...")
        val_scores = validate(model, val_loader, device, logger, epoch)
        
        f1_score = val_scores.get('F1_1', 0.0)
        logger.info(f"Epoch {epoch} Validation Results:")
        logger.info(f"  F1-Score: {f1_score:.6f} ({f1_score*100:.2f}%)")
        logger.info(f"  Precision: {val_scores.get('precision_1', 0.0):.6f}")
        logger.info(f"  Recall: {val_scores.get('recall_1', 0.0):.6f}")
        logger.info(f"  IoU: {val_scores.get(1, 0.0):.6f}")
        logger.info(f"  Overall Acc: {val_scores.get('Overall_Acc', 0.0):.6f}")
        
        # Save best model
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epoch = epoch
            checkpoint_path = os.path.join(output_dir, 'best_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': f1_score,
                'val_scores': val_scores
            }, checkpoint_path)
            logger.info(f"Saved best model (F1: {f1_score:.6f}) to {checkpoint_path}")
    
    logger.info("="*60)
    logger.info("Fine-tuning complete!")
    logger.info(f"Best F1: {best_f1:.6f} at epoch {best_epoch}")
    logger.info(f"Checkpoint saved to: {os.path.join(output_dir, 'best_checkpoint.pth')}")
    logger.info("="*60)
    
    # Test on test set
    logger.info("\nEvaluating on test set...")
    test_opt = copy.deepcopy(opt_stanet)
    test_opt.dataroot = str(test_dir)
    test_opt.dataset_mode = 'changedetection'
    test_opt.batch_size = args.batch_size
    test_opt.num_threads = 0
    test_opt.serial_batches = True
    test_opt.preprocess = ''
    test_opt.angle = 0
    test_opt.no_flip = True
    test_opt.phase = 'val'  # Use 'val' to get labels
    test_opt.isTrain = False
    
    test_dataset = create_dataset(test_opt)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_checkpoint.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['f1_score']:.6f}")
    
    test_scores = validate(model, test_dataset, device, logger, 'test')
    
    logger.info("="*60)
    logger.info("Test Set Results:")
    logger.info(f"  F1-Score: {test_scores.get('F1_1', 0.0):.6f} ({test_scores.get('F1_1', 0.0)*100:.2f}%)")
    logger.info(f"  Precision: {test_scores.get('precision_1', 0.0):.6f}")
    logger.info(f"  Recall: {test_scores.get('recall_1', 0.0):.6f}")
    logger.info(f"  IoU: {test_scores.get(1, 0.0):.6f}")
    logger.info(f"  Overall Acc: {test_scores.get('Overall_Acc', 0.0):.6f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
