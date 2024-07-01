from mmengine.registry import Registry
from mmdet.models.detectors import FasterRCNN

# Create a custom registry for models
MY_MODELS = Registry('my_model')
# Register FasterRCNN into the custom registry
MY_MODELS.register_module(module=FasterRCNN)
