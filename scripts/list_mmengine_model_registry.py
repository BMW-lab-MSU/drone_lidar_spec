from mmengine.registry import MODELS

# Import specific modules to ensure they are registered
from mmdet.models.detectors import FasterRCNN

def list_registered_models():
    print("Registered models in the mmengine::model registry:")
    if not MODELS.module_dict:
        print("No models registered. There might be an issue with model registration.")
    for model_name in sorted(MODELS.module_dict.keys()):
        print(model_name)

if __name__ == "__main__":
    list_registered_models()
