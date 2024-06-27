import os
import mmdet

# Get the path to the mmdet package
mmdet_path = os.path.dirname(mmdet.__file__)

# Construct the path to the base config file
base_config_path = os.path.join(mmdet_path, '../configs/_base_/models/faster_rcnn_r50_fpn.py')

_base_ = [base_config_path]

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
