_base_ = ['../common/ms_3x_coco.py', '../_base_/models/faster-rcnn_r50_fpn.py']

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
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[9, 11],
        type='MultiStepLR'),
]

resume = False

test_cfg = dict(type='TestLoop')

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')

train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)

train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train2017.json',
            backend_args=None,
            data_prefix=dict(img='train2017/'),
            data_root='data/coco/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=3,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

val_cfg = dict(type='ValLoop')

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
)
