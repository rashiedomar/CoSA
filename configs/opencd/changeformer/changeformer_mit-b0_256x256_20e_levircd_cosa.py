_base_ = [
    '../_base_/models/changeformer_mit-b0.py',
    '../common/standard_256x256_40k_levircd.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

# Enable CoSA in FeatureFusionNeck (x4 stage)
model = dict(
    pretrained=checkpoint,
    neck=dict(
        type='FeatureFusionNeck',
        policy='concat',
        in_channels=[32, 64, 160, 256],
        use_cosa=True,
        cosa_index=0,
        cosa_topk=32,
        cosa_gamma_init=1.0,
        cosa_use_learnable_gate=True),
    decode_head=dict(num_classes=2))

# Fine-tune schedule: 20 epochs ~= 1120 iters (batch 8, 445 train)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1120, val_interval=1120)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', power=1.0, begin=100, end=1120, eta_min=0.0, by_epoch=False)
]

# Fine-tune optimizer (keep small LR, boost CoSA)
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'cosa_block': dict(lr_mult=10., decay_mult=0.),
        }))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1120, save_best='mIoU'))
