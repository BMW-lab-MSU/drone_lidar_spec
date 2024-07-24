_base_ = ['../_base_/models/faster-rcnn_r50_fpn.py']

dataset_type = 'CocoDataset'
classes = ('drone_frequency', )

normalization_values = {
    'mean': [128.41, 205.31, 76.05],
    'std': [52.56, 22.05, 31.79]
}

# DataLoader settings for training
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/train/annotations/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/train/images/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

# DataLoader settings for validation
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/val/annotations/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/val/images/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

# DataLoader settings for testing
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/test/annotations/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/test/images/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

# Evaluator settings
val_evaluator = dict(ann_file='placeholder/val/annotations/annotations.json')  # Placeholder to be replaced by cfg-options
test_evaluator = dict(ann_file='placeholder/test/annotations/annotations.json')  # Placeholder to be replaced by cfg-options

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=8,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # Keep all stages
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/d86p233/Desktop/BMW-spec/mmdetection/checkpoints/faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco_20210604_182954-002e082a.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # Include all stages
        out_channels=256,
        num_outs=5),  # Keep original outputs
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[5, 5, 5],  # Adjusted scale
            ratios=[31],  # Adjusted ratio for the desired anchor size
            strides=[5, 5, 5, 5, 5]),  # Use consistent stride to match the scale
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[5, 5, 5, 5]),  # Use consistent stride
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # Adjusted to 1 class
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_begin=1,
    val_interval=1
)

# Optimization wrapper configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        paramwise_cfg=dict(
            custom_keys={'backbone': dict(lr_mult=0.1)}
        )
    )
)

# Learning rate configuration without warmup
lr_config = dict(
    policy='step',
    step=[8, 11],
    gamma=0.1
)

# Checkpoint settings
checkpoint_config = dict(
    interval=1
)

# Logging settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Evaluation settings
evaluation = dict(
    interval=1,
    metric='bbox'
)

# Workflow settings
workflow = [('train', 1), ('val', 1)]
