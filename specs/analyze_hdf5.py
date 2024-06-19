import h5py
import os
import argparse

def inspect_hdf5_group(group, level=0):
    """Recursively print the structure of an HDF5 group."""
    indent = "  " * level
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {key}")
            inspect_hdf5_group(item, level + 1)
        else:
            print(f"{indent}Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")

def analyze_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        print(f"Inspecting structure of {file_path}:")
        inspect_hdf5_group(hdf5_file)
        
        # Attempt to access the specific dataset
        try:
            dataset_path = 'parameters/prop_frequency/front_right/avg'
            print(f"Attempting to access dataset at path: {dataset_path}")
            data = hdf5_file[dataset_path][0]
            print(f"Dataset '{dataset_path}' contents:\n{data}")
        except KeyError as e:
            print(f"KeyError: {e}")
        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze the structure of the first HDF5 file in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing HDF5 files.")

    args = parser.parse_args()

    input_folder = args.input_folder

    # Find the first HDF5 file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            file_path = os.path.join(input_folder, filename)
            analyze_hdf5_file(file_path)
            break
    else:
        print("No HDF5 files found in the specified folder.")

if __name__ == "__main__":
    main()
