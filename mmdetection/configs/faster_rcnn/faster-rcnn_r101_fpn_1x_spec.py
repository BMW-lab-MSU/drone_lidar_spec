_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

# Dataset type and root
dataset_type = 'CocoDataset'
data_root = '/home/d86p233/Desktop/BMW-spec/specs/single_freq_raw_specs/'
classes = ('drone_frequency', )

# Normalization values
normalization_values = {
    'mean': [44.34, 125.08, 138.27],
    'std': [26.87, 26.33, 14.68]
}

# Train dataloader
train_dataloader = dict(
    batch_size=2,  # Number of images per batch per GPU
    num_workers=2,  # Number of CPU workers to load data for each GPU
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/Raw/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')  # Remove 'keys' argument
        ]
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=2,  # Number of images per batch per GPU
    num_workers=2,  # Number of CPU workers to load data for each GPU
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/Raw/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')  # Remove 'keys' argument
        ]
    )
)

# Test dataloader
test_dataloader = dict(
    batch_size=2,  # Number of images per batch per GPU
    num_workers=2,  # Number of CPU workers to load data for each GPU
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/Raw/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')  # Remove 'keys' argument
        ]
    )
)

# Evaluators
val_evaluator = dict(ann_file=data_root + 'val/annotations.json')
test_evaluator = dict(ann_file=data_root + 'test/annotations.json')

# Model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=8,
        init_cfg=dict(type='Pretrained', checkpoint='/home/d86p233/Desktop/BMW-spec/mmdetection/checkpoints/resnext101_32x8d-110c445d.pth')
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1  # Adjust the number of classes as needed
        )
    )
)

# Optimizer configuration
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}  # Lower learning rate for backbone layers
    )
)

# Learning rate scheduler
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],  # Learning rate decay steps
    gamma=0.1  # Learning rate decay factor
)

# Checkpoint configuration
checkpoint_config = dict(
    interval=1  # Interval for saving checkpoints
)

# Logging configuration
log_config = dict(
    interval=50,  # Interval for logging
    hooks=[
        dict(type='TextLoggerHook'),  # Log to text
        dict(type='TensorboardLoggerHook')  # Log to Tensorboard
    ]
)

# Evaluation configuration
evaluation = dict(
    interval=1,  # Interval for evaluation
    metric='bbox'  # Metric for evaluation
)

# Workflow
workflow = [('train', 1), ('val', 1)]
