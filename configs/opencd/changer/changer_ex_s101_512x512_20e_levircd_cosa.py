_base_ = './changer_ex_s101_512x512_40k_levircd.py'

# Enable CoSA in Changer decode head
model = dict(
    decode_head=dict(
        use_cosa=True,
        cosa_topk=32,
        cosa_gamma_init=0.0,
        cosa_use_learnable_gate=True,
        freeze_except_cosa=True))

# Fine-tune schedule: 20 epochs ~= 1120 iters (batch 8, 445 train)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1120, val_interval=1120)

# Override scheduler to match short fine-tune
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', power=1.0, begin=100, end=1120, eta_min=0.0, by_epoch=False)
]

# Slightly lower LR for fine-tuning
optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# Save only at end
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1120, save_best='mIoU'))
