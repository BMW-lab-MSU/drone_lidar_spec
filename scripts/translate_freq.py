import json

def bbox_to_frequency(bbox, img_height=462, freq_range=1952):
    """
    Retranslate the bounding box coordinates back to the frequency.

    Parameters:
    - bbox (list): Bounding box coordinates in the format [x, y, width, height].
    - img_height (int): Height of the image (default is 462).
    - freq_range (int): The range of frequencies (default is 1952).

    Returns:
    - frequency (float): The retranslated frequency.
    """
    # Extract the y coordinate and height of the bbox
    y = bbox[1]
    height = bbox[3]
    
    # Calculate the center of the bbox in the image coordinate
    bbox_y_center = img_height - (y + height / 2)
    
    # Retranslate the bbox center to the frequency
    frequency = (bbox_y_center / img_height) * freq_range
    
    return frequency

def main():
    # Example input JSON file path
    json_file_path = 'annotations.json'
    
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Process each annotation and retranslate the bbox to frequency
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        frequency = bbox_to_frequency(bbox)
        annotation['frequency'] = frequency
        print(f"Annotation ID: {annotation['id']}, Frequency: {frequency:.2f} Hz")

if __name__ == "__main__":
    main()
