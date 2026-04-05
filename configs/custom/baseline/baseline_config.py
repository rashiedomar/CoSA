"""
Baseline Siamese U-Net Configuration
Our custom baseline model for change detection.

Model: SiameseUNet
- Architecture: FC-Siam-diff style
- Encoder: Shared Siamese encoder
- Fusion: Absolute difference
- Decoder: U-Net style decoder
"""

# Model configuration
model = dict(
    type='SiameseUNet',
    in_channels=3,
    n_classes=1,
    base_channels=64,
    fusion='diff'  # Absolute difference fusion
)

# Training configuration
train_cfg = dict(
    batch_size=8,
    epochs=100,
    learning_rate=0.0001,
    optimizer='AdamW',
    weight_decay=1e-5,
    loss='BCEWithLogitsLoss',
    pos_weight=1.0,
    seed=42
)

# Dataset configuration
dataset = dict(
    name='LEVIR-CD',
    root_dir='datasets/to_check_dataset/LEVIR-CD_combined',
    train_size=445,
    val_size=64,
    test_size=128,
    base_size=256,
    augment=True
)

# Model parameters
total_parameters = 33_233_633

