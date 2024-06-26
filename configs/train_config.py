# Optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}  # Lower learning rate for backbone layers
    )
)

# Learning Rate Scheduler
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],  # Learning rate decay steps
    gamma=0.1  # Learning rate decay factor
)

# Training Runner Configuration
runner = dict(
    type='EpochBasedRunner',
    max_epochs=12  # Number of epochs to train
)

# Checkpoint Configuration
checkpoint_config = dict(
    interval=1  # Interval for saving checkpoints
)

# Logging Configuration
log_config = dict(
    interval=50,  # Interval for logging
    hooks=[
        dict(type='TextLoggerHook'),  # Log to text
        dict(type='TensorboardLoggerHook')  # Log to Tensorboard
    ]
)

# Custom Hooks
custom_hooks = [
    dict(type='NumClassCheckHook'),  # Check if the number of classes matches the dataset
    dict(
        type='UnfreezeLayersHook',  # Gradual unfreezing schedule
        unfreeze_schedule={2: 5, 1: 8, 0: 10}
    )
]

# Evaluation Configuration
evaluation = dict(
    interval=1,  # Interval for evaluation
    metric='bbox'  # Metric for evaluation
)

# Distributed Training Parameters
dist_params = dict(backend='nccl')

# Logging Level
log_level = 'INFO'

# Load From (Pretrained Model)
load_from = None  # Set to checkpoint path if resuming from a specific checkpoint

# Resume From (Checkpoint to Resume From)
resume_from = None

# Workflow
workflow = [('train', 1), ('val', 1)]

# Work Directory
work_dir = './work_dirs/faster_rcnn_resnext101'

# GPU Settings
gpu_ids = range(0, 2)  # Use GPUs with IDs 0 and 1
