_base_ = [
    './faster-rcnn_r50_fpn.py'
]

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
