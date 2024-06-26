dataset_type = 'CocoDataset'
data_root = 'specs/split_specs/'

# File paths for training, validation, and testing datasets
train_ann_file = data_root + 'train/annotations.json'
train_img_prefix = data_root + 'train/Raw/'

val_ann_file = data_root + 'val/annotations.json'
val_img_prefix = data_root + 'val/Raw/'

test_ann_file = data_root + 'test/annotations.json'
test_img_prefix = data_root + 'test/Raw/'

# Normalization values
# UPDATED THESE WITH CALCULATED VALUES
norm_mean = [123.675, 116.28, 103.53]
norm_std = [58.395, 57.12, 57.375]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,  # Train annotations file
        img_prefix=train_img_prefix,  # Raw images for training
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=norm_mean, std=norm_std, to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,  # Validation annotations file
        img_prefix=val_img_prefix,  # Raw images for validation
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=norm_mean, std=norm_std, to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    ),
    test=dict(
        type=dataset_type,
        ann_file=test_ann_file,  # Test annotations file
        img_prefix=test_img_prefix,  # Raw images for testing
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=norm_mean, std=norm_std, to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
)

evaluation = dict(interval=1, metric='bbox')
