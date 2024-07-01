#import mmcv
#import mmdet
from mmdet.models import *  # Import all models to ensure registration
from mmengine.registry import MODELS

def list_registered_models():
    print("Registered models in the mmengine::model registry:")
    if not MODELS.module_dict:
        print("No models registered.")
    for model_name in MODELS.module_dict.keys():
        print(model_name)

if __name__ == "__main__":
    list_registered_models()
