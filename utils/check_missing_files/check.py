import pandas as pd
import argparse
import os
import h5py

def read_h5_file(file_path):
    """
    Read a H5 file and return its contents
    Args:
        file_path (str): Path to the H5 file
    Returns:
        dict: Contents of the H5 file or None if file cannot be read
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Create a dictionary to store all datasets
            contents = {}
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    contents[name] = obj[()]
            
            # Visit all groups and datasets in the file
            f.visititems(visit_func)
            return contents
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        images = df['image'].to_numpy()
        images = [img.replace('.tif', '.h5') for img in images]
        return images
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, help='Path to the features folder')
    args = parser.parse_args()

    print("features_path", args.features_path)
    
    file_path = "./reference.csv"
    folder_features_path = args.features_path
    features_list = read_csv_file(file_path)

    # Get list of files in features directory
    available_files = os.listdir(folder_features_path)

    # Check which files from features_list exist
    missing_files = []
    for feature_file in features_list:
        if feature_file not in available_files:
            missing_files.append(feature_file)

    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(file)
        print(f"\nTotal missing files: {len(missing_files)}")
    else:
        print("\nAll files exist in features folder")
