model = dict(
    type='FasterRCNN',  # Model type is Faster R-CNN
    backbone=dict(
        type='ResNeXt',  # Backbone type is ResNeXt
        depth=101,  # Depth of the ResNeXt model
        groups=32,  # Number of groups in ResNeXt
        base_width=4,  # Base width in ResNeXt
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnext101_32x4d')  # Use pretrained weights from ImageNet
    ),
    neck=dict(
        type='FPN',  # Neck type is FPN (Feature Pyramid Network)
        in_channels=[256, 512, 1024, 2048],  # Input channels from the backbone feature maps
        out_channels=256,  # Output channels of FPN
        num_outs=5  # Number of output scales from the FPN
    ),
    rpn_head=dict(
        type='RPNHead',  # The head used for generating region proposals
        in_channels=256,  # Input channels from FPN
        feat_channels=256,  # Feature channels in the RPN head
        anchor_generator=dict(
            type='AnchorGenerator',  # The type of anchor generator
            scales=[8],  # Anchor scales
            ratios=[0.5, 1.0, 2.0],  # Anchor aspect ratios
            strides=[4, 8, 16, 32, 64]  # Anchor strides
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0  # Classification loss for RPN
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)  # Bounding box regression loss for RPN
    ),
    roi_head=dict(
        type='StandardRoIHead',  # The head used for processing region proposals and classifying objects
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # The type of ROI extractor
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),  # ROI align settings
            out_channels=256,  # Output channels from ROI extractor
            featmap_strides=[4, 8, 16, 32]  # Feature map strides
        ),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',  # The type of bounding box head
            in_channels=256,  # Input channels to the bounding box head
            fc_out_channels=1024,  # Output channels from the fully connected layers
            roi_feat_size=7,  # ROI feature size
            num_classes=1,  # Number of object classes; set to 1 for single class
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',  # Bounding box coder type
                target_means=[0., 0., 0., 0.],  # Target means for bounding box regression
                target_stds=[0.1, 0.1, 0.2, 0.2]  # Target standard deviations for bounding box regression
            ),
            reg_class_agnostic=False,  # Whether the regression is class-agnostic
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0  # Classification loss for ROI head
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)  # Bounding box regression loss for ROI head
        )
    )
)
