_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnext101_32x4d')
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,  # Adjust the number of classes as needed
        )
    )
)

# Dataset type and root
dataset_type = 'CocoDataset'
data_root = '/home/d86p233/Desktop/BMW-spec/specs/single_freq_raw_specs/'

# Annotation and image paths
train_ann_file = data_root + 'train/annotations.json'
train_img_prefix = data_root + 'train/Raw/'

val_ann_file = data_root + 'val/annotations.json'
val_img_prefix = data_root + 'val/Raw/'

test_ann_file = data_root + 'test/annotations.json'
test_img_prefix = data_root + 'test/Raw/'

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
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='DefaultFormatBundle'),  # Processes annotations (such as converting to tensor)
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # Prepares data to feed into the model
        ]
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=2,  # Number of images per batch per GPU
    num_workers=2,  # Number of CPU workers to load data for each GPU
    dataset=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    )
)

# Test dataloader
test_dataloader = dict(
    batch_size=2,  # Number of images per batch per GPU
    num_workers=2,  # Number of CPU workers to load data for each GPU
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
)

# Evaluators
val_evaluator = dict(ann_file=val_ann_file)
test_evaluator = dict(ann_file=test_ann_file)

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
# How many epochs to train before validation
# This specifies to train for 1 then val for 1, every other
workflow = [('train', 1), ('val', 1)]
