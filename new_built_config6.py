checkpoint_config = dict(interval=1)
classes = ('drone_frequency', )
dataset_type = 'CocoDataset'
evaluation = dict(interval=1, metric='bbox')
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50)
lr_config = dict(
    gamma=0.1, policy='step', step=[
        8,
        11,
    ])
model = dict(
    backbone=dict(
        base_width=8,
        depth=101,
        frozen_stages=1,
        groups=32,
        init_cfg=dict(
            checkpoint=
            '/home/d86p233/Desktop/BMW-spec/mmdetection/checkpoints/faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco_20210604_182954-002e082a.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=80,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                5,
                5,
                5,
                5,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                31,
            ],
            scales=[
                5,
                5,
                5,
            ],
            strides=[
                5,
                5,
                5,
                5,
                5,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
normalization_values = dict(
    mean=[
        128.41,
        205.31,
        76.05,
    ], std=[
        52.56,
        22.05,
        31.79,
    ])
optimizer = dict(
    lr=0.01,
    momentum=0.9,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='SGD',
    weight_decay=0.0001)
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='placeholder/test/annotations.json',
        data_prefix=dict(img='placeholder/test/'),
        data_root='placeholder',
        metainfo=dict(classes=('drone_frequency', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                775,
                462,
            ), type='Resize'),
            dict(
                mean=[
                    128.41,
                    205.31,
                    76.05,
                ],
                std=[
                    52.56,
                    22.05,
                    31.79,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2)
test_evaluator = dict(ann_file='placeholder/test/annotations.json')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='placeholder/train/annotations.json',
        data_prefix=dict(img='placeholder/train/'),
        data_root='placeholder',
        metainfo=dict(classes=('drone_frequency', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                775,
                462,
            ), type='Resize'),
            dict(
                mean=[
                    128.41,
                    205.31,
                    76.05,
                ],
                std=[
                    52.56,
                    22.05,
                    31.79,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2)
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='placeholder/val/annotations.json',
        data_prefix=dict(img='placeholder/val/'),
        data_root='placeholder',
        metainfo=dict(classes=('drone_frequency', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                775,
                462,
            ), type='Resize'),
            dict(
                mean=[
                    128.41,
                    205.31,
                    76.05,
                ],
                std=[
                    52.56,
                    22.05,
                    31.79,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2)
val_evaluator = dict(ann_file='placeholder/val/annotations.json')
workflow = [
    (
        'train',
        1,
    ),
    (
        'val',
        1,
    ),
]
