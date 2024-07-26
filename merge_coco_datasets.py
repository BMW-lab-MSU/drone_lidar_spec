import os
import shutil
import json

# Base path to the data directories
base_dir = "/home/d86p233/Desktop/BMW-spec/data"
data_dirs = [
    "boost_all", "boost_all_SF", "boost_full", "boost_full_SF",
    "boost_partial", "boost_partial_SF", "boost_prop-only",
    "boost_prop-only-90", "noboost_all", "noboost_all_SF",
    "noboost_full", "noboost_full_SF", "noboost_partial",
    "noboost_partial_SF", "noboost_prop-only", "noboost_prop-only-90"
]

# Destination base path
merged_data_base_dir = "/home/d86p233/Desktop/BMW-spec/merged_data"

# Function to merge annotations for a specific subfolder
def merge_annotations(data_dir, output_dir):
    print(f"Processing annotations for {data_dir}")
    merged_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    image_id_offset = 0
    annotation_id_offset = 0

    for split in ["train", "test", "val"]:
        ann_file = os.path.join(data_dir, split, "annotations", "annotations.json")
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                data = json.load(f)
                # Add categories only once
                if not merged_annotations["categories"]:
                    merged_annotations["categories"] = data["categories"]
                # Update image and annotation IDs
                for image in data["images"]:
                    image["id"] += image_id_offset
                    merged_annotations["images"].append(image)
                for annotation in data["annotations"]:
                    annotation["id"] += annotation_id_offset
                    annotation["image_id"] += image_id_offset
                    merged_annotations["annotations"].append(annotation)
            # Update offsets
            image_id_offset += len(data["images"])
            annotation_id_offset += len(data["annotations"])

    # Save merged annotations
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(merged_annotations, f)
    print(f"Merged annotations saved to {os.path.join(output_dir, 'annotations.json')}")

# Function to merge images for a specific subfolder
def merge_images(data_dir, output_dir):
    print(f"Processing images for {data_dir}")
    os.makedirs(output_dir, exist_ok=True)
    image_count = 0
    for split in ["train", "test", "val"]:
        images_dir = os.path.join(data_dir, split, "images")
        if os.path.exists(images_dir):
            for image_file in os.listdir(images_dir):
                src = os.path.join(images_dir, image_file)
                dst = os.path.join(output_dir, image_file)
                shutil.copy(src, dst)
                image_count += 1
                if image_count % 500 == 0:
                    print(f"{image_count} images processed for {data_dir}")
    print(f"Total {image_count} images processed for {data_dir}")

# Process each subfolder
for subfolder in data_dirs:
    subfolder_path = os.path.join(base_dir, subfolder)
    merged_annotations_subfolder_path = os.path.join(merged_data_base_dir, subfolder, "annotations")
    merged_images_subfolder_path = os.path.join(merged_data_base_dir, subfolder, "images")

    merge_annotations(subfolder_path, merged_annotations_subfolder_path)
    merge_images(subfolder_path, merged_images_subfolder_path)

print("Merging completed.")
