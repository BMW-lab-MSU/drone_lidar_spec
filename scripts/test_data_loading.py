from mmengine.config import Config
from mmengine.registry import DATASETS
from mmdet.datasets import CocoDataset

# Explicitly register CocoDataset (if not already registered)
#DATASETS.register_module(CocoDataset)

# Load the configuration file
cfg = Config.fromfile('/home/d86p233/Desktop/BMW-spec/work_dirs/train_task_0/20240724_082412/vis_data/config.py')

# Print all registered datasets in the mmengine dataset registry
print("Registered datasets in mmengine::dataset registry:")
for dataset_name in DATASETS.module_dict:
    print(dataset_name)

# Attempt to build the dataset
try:
    train_dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    print('Number of samples in the dataset:', len(train_dataset))
except KeyError as e:
    print(f"Error building dataset: {e}")
