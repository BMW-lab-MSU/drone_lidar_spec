dataset_type = 'CocoDataset'
data_root = 'spectrograms/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '{}/annotations.json',  # Annotations relative to each dataset folder
        img_prefix=data_root + '{}/Labeled/',       # Labeled images relative to each dataset folder
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '{}/annotations.json',
        img_prefix=data_root + '{}/Labeled/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '{}/annotations.json',
        img_prefix=data_root + '{}/Raw/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
)

evaluation = dict(interval=1, metric='bbox')
