_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
classes = ('drone_frequency', )

normalization_values = {
    'mean': [128.41, 205.31, 76.05],
    'std': [52.56, 22.05, 31.79]
}

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/train/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/train/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/val/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/val/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root='placeholder',  # Placeholder to be replaced by cfg-options
        ann_file='placeholder/test/annotations.json',  # Placeholder to be replaced by cfg-options
        data_prefix=dict(img='placeholder/test/'),  # Placeholder to be replaced by cfg-options
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(775, 462)),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(ann_file='placeholder/val/annotations.json')  # Placeholder to be replaced by cfg-options
test_evaluator = dict(ann_file='placeholder/test/annotations.json')  # Placeholder to be replaced by cfg-options

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
            num_classes=1
        )
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2.5, 1.25, 0.625, 0.3125, 0.15625],  # Scales adjusted to maintain 20px height
            ratios=[38.75],  # Single aspect ratio
            strides=[8, 16, 32, 64, 128]  # Corresponding feature map strides
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    )
)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
    gamma=0.1
)

checkpoint_config = dict(
    interval=1
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

evaluation = dict(
    interval=1,
    metric='bbox'
)

workflow = [('train', 1), ('val', 1)]
