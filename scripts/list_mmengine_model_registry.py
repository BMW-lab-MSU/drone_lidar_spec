import mmcv
import mmdet
from mmengine.registry import Registry

# Create a custom registry for models
MY_MODELS = Registry('my_model')

# Import the necessary functions and modules to trigger model registration
from mmdet.models.detectors import FasterRCNN

# Register FasterRCNN into the custom registry
MY_MODELS.register_module(module=FasterRCNN)

def list_registered_models():
    print("Registered models in the my_model registry:")
    if not MY_MODELS.module_dict:
        print("No models registered. There might be an issue with model registration.")
    for model_name in sorted(MY_MODELS.module_dict.keys()):
        print(model_name)

if __name__ == "__main__":
    list_registered_models()
