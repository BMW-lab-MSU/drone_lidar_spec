dataset_type = 'CocoDataset'
data_root = 'data/spectrograms/'

data = dict(
    samples_per_gpu=2,  # Number of samples per GPU (adjust based on your GPU memory)
    workers_per_gpu=2,  # Number of data loading workers per GPU
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',  # Path to the COCO-format annotations file
        img_prefix=data_root + 'Labeled/',  # Path to labeled images
        pipeline=[  # Data processing pipeline for training
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',  # Path to the COCO-format annotations file
        img_prefix=data_root + 'Labeled/',  # Path to labeled images
        pipeline=[  # Data processing pipeline for validation/testing
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.json',  # Path to the COCO-format annotations file
        img_prefix=data_root + 'Raw/',  # Path to raw images for testing
        pipeline=[  # Data processing pipeline for testing
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
)

evaluation = dict(interval=1, metric='bbox')
