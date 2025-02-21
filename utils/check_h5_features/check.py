import os
import argparse
import h5py
import numpy as np

def count_files_in_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return len(files)
    except Exception as e:
        print(f"Error counting files: {e}")
        return -1


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
    parser.add_argument('--patch_folder_path', type=str, help='Path to the patch folder')
    parser.add_argument('--h5_folder_path', type=str, help='Path to the h5 folder')

    args = parser.parse_args()

    wsi_names = []

    # List all folders in patch folder
    if args.patch_folder_path:
        try:
            patch_folders = [d for d in os.listdir(args.patch_folder_path) 
                           if os.path.isdir(os.path.join(args.patch_folder_path, d))]
            for folder in patch_folders:
                wsi_names.append(folder)
        except Exception as e:
            print(f"Error listing patch folders: {e}")

    for wsi_name in wsi_names:
        print("---------------------------------")
        print(f'wsi_name: {wsi_name}')
        h5_path = os.path.join(args.h5_folder_path, f"{wsi_name}.h5")
        if os.path.exists(h5_path):
            num_features = get_number_of_features(h5_path)
            num_images = count_files_in_folder(os.path.join(args.patch_folder_path, wsi_name))
            if num_features != num_images:
                print(f"Number of features: {num_features} != Number of images: {num_images}")
        else:
            print(f"Missing H5 file for {wsi_name}")
            
    # file_count = count_files_in_folder(args.folder_path)
    # if file_count >= 0:
    #     print(f"Number of files in the folder: {file_count}")

    # data = get_number_of_features(args.file_path)
    # print(f"Number of features: {data}")