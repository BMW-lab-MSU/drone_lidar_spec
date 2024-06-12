import os
import fiftyone as fo
import fiftyone.zoo as foz

# Specify the output directory (this will be the dataset_dir)
output_dir = "mmdetection/custom_imgs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Specify the dataset and split
dataset_name = "coco-2017"
split = "validation"

# Download the dataset directly to the output directory
dataset = foz.load_zoo_dataset(
    dataset_name,
    split=split,
    max_samples=100,
    dataset_dir=output_dir,
    label_types=[],  # Empty list means no labels will be downloaded, only images
)

print(f"Downloaded {len(dataset)} images to {output_dir}")
