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

    # Print the results
    print(results)

if __name__ == '__main__':
    main()
