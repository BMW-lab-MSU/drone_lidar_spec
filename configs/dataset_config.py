dataset_type = 'CocoDataset'
data_root = 'specs/split_specs/'

train_ann_file = data_root + 'train/annotations.json'
train_img_prefix = data_root + 'train/Raw/'

val_ann_file = data_root + 'val/annotations.json'
val_img_prefix = data_root + 'val/Raw/'

test_ann_file = data_root + 'test/annotations.json'
test_img_prefix = data_root + 'test/Raw/'

normalization_values = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375]
}

data = dict(
    samples_per_gpu=2, #How many images per batch
    workers_per_gpu=2, #Number of CPU workers to load data for each GPU
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Normalize', **normalization_values, to_rgb=True),
            dict(type='DefaultFormatBundle'), #Processes annotations (such as converting to tensor)
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), #Prepares data to feed into modellll
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
