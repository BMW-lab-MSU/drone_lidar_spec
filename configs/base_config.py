model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='BaseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnext101_32x4d')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # Adjust the number of classes as needed
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}  # Lower learning rate for backbone layers
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],  # Learning rate decay steps
    gamma=0.1  # Learning rate decay factor
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=12  # Number of epochs to train
)

checkpoint_config = dict(
    interval=1  # Interval for saving checkpoints
)

log_config = dict(
    interval=50,  # Interval for logging
    hooks=[
        dict(type='TextLoggerHook'),  # Log to text
        dict(type='TensorboardLoggerHook')  # Log to Tensorboard
    ]
)

custom_hooks = [
    dict(type='NumClassCheckHook'),  # Check if the number of classes matches the dataset
    dict(
        type='UnfreezeLayersHook',  # Gradual unfreezing schedule
        unfreeze_schedule={2: 5, 1: 8, 0: 10}
    )
]

evaluation = dict(
    interval=1,  # Interval for evaluation
    metric='bbox'  # Metric for evaluation
)

dist_params = dict(backend='nccl')

log_level = 'INFO'

load_from = None  # Set to checkpoint path if resuming from a specific checkpoint

resume_from = None

workflow = [('train', 1), ('val', 1)]

work_dir = './work_dirs/faster_rcnn_resnext101'

gpu_ids = range(0, 2)  # Use GPUs with IDs 0 and 1

dataset_type = 'CocoDataset'
data_root = '/home/d86p233/Desktop/BMW-spec/specs/single_freq_raw_specs/'

train_ann_file = data_root + 'train/annotations.json'
train_img_prefix = data_root + 'train/Raw/'

val_ann_file = data_root + 'val/annotations.json'
val_img_prefix = data_root + 'val/Raw/'

test_ann_file = data_root + 'test/annotations.json'
test_img_prefix = data_root + 'test/Raw/'

normalization_values = {
    'mean': [44.34, 125.08, 138.27],
    'std': [26.87, 26.33, 14.68]
}

data = dict(
    samples_per_gpu=2,  # How many images per batch
    workers_per_gpu=2,  # Number of CPU workers to load data for each GPU
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='DefaultFormatBundle'),  # Processes annotations (such as converting to tensor)
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # Prepares data to feed into model
        ]
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    ),
    test=dict(
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
