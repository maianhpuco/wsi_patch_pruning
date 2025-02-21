import os
import argparse
import h5py
import numpy as np


def get_number_of_features(file_path):
    """
    Read data from an H5 file containing features, patch indices, coordinates, and labels.
    
    Args:
        file_path (str): Path to the H5 file
        
    Returns:
        dict: Dictionary containing the datasets from the H5 file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Read all datasets into memory
            data = {
                'features': np.array(f['features']),
                'patch_indices': np.array(f['patch_indices']),
                'coordinates': np.array(f['coordinates']),
                'label': np.array(f['label']),
                'spixel_idx': np.array(f['spixel_idx'])
            }
            
            
            return data['features'].shape[0]
            
    except Exception as e:
        print(f"Error reading H5 file {file_path}: {e}")
        return None


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count files in a folder')
    parser.add_argument('--h5_folder_path', type=str, help='Path to the h5 folder')

    args = parser.parse_args()

    # List all h5 files in the folder
    wsi_names = []
    try:
        h5_files = [f.split('.')[0] for f in os.listdir(args.h5_folder_path) 
                   if f.endswith('.h5') and os.path.isfile(os.path.join(args.h5_folder_path, f))]
        wsi_names.extend(h5_files)
    except Exception as e:
        print(f"Error listing h5 files: {e}")

    for wsi_name in wsi_names:
        num_features = get_number_of_features(wsi_name)
        print(f'{wsi_name},{num_features}')
    # file_count = count_files_in_folder(args.folder_path)
    # if file_count >= 0:
    #     print(f"Number of files in the folder: {file_count}")

    # data = get_number_of_features(args.file_path)
    # print(f"Number of features: {data}")