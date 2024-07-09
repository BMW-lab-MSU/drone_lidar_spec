import json

annotation_file = '/home/d86p233/Desktop/BMW-spec/all_specs_sorted/all/train/annotations.json'

with open(annotation_file, 'r') as f:
    data = json.load(f)

print("Number of images:", len(data['images']))
