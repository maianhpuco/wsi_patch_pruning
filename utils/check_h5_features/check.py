import os
import argparse

def count_files_in_folder(folder_path):
    """
    Count the number of files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        int: Number of files in the folder
    """
    try:
        # Get list of all files in the directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return len(files)
    except Exception as e:
        print(f"Error counting files: {e}")
        return -1

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count files in a folder')
    parser.add_argument('folder_path', required=True, type=str, help='Path to the folder to count files in')
    
    args = parser.parse_args()
    file_count = count_files_in_folder(args.folder_path)
    if file_count >= 0:
        print(f"Number of files in the folder: {file_count}")
