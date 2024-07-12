import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Load and print results from a pickle file.')
    parser.add_argument('output_file', help='Path to the output pickle file')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.output_file, 'rb') as f:
        results = pickle.load(f)

    # Check for detected bounding boxes
    detected_images = []
    for result in results:
        img_path = result['img_path']
        bboxes = result['pred_instances']['bboxes']
        if bboxes.size(0) > 0:
            detected_images.append(img_path)

    if detected_images:
        print("Bounding boxes were detected in the following images:")
        for img in detected_images:
            print(img)
    else:
        print("No bounding boxes were detected in any images.")

if __name__ == '__main__':
    main()
