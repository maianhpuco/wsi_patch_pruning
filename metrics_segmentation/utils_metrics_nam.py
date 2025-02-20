import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import h5py
from tqdm import tqdm 

import numpy as np
from tqdm import tqdm
from rtree import index  # R-tree for efficient spatial queries
 
# xml_folder = (
#     "/Users/nam.le/Desktop/research/wsi_patch_pruning/metrics_segmentation/data"
# )
# list_xml_file = [os.path.join(xml_folder, name) for name in os.listdir(xml_folder)]


def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)  # Load XML file
        root = tree.getroot()  # Get root element
        return root
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def extract_coordinates(file_path):
    all_coordinates = []
    root = parse_xml(file_path)
    if root is None:
        return  # Skip if parsing failed

    print(f"Processing XML: {file_path}")

    # Loop through all 'Coordinate' elements
    for coordinate in root.findall(".//Coordinate"):
        order = coordinate.attrib.get("Order")
        x = coordinate.attrib.get("X")
        y = coordinate.attrib.get("Y")

        # Append extracted data
        if order and x and y:
            all_coordinates.append(
                {
                    "File": file_path.split("/")[-1],  # Extract only the filename
                    "Order": int(order),
                    "X": float(x),
                    "Y": float(y),
                }
            )
    all_coordinates = pd.DataFrame(all_coordinates)
    return all_coordinates


def return_df_xml(xml_path):
    return extract_coordinates(xml_path)



def read_h5_data(file_path, dataset_name=None):
    data = None
    with h5py.File(file_path, "r") as file:
        if dataset_name is not None:
            if dataset_name in file:
                dataset = file[dataset_name]
                data = dataset[()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
        else:
            datasets = {}

            def visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets[name] = node[()]

            file.visititems(visitor)

            if len(datasets) == 1:
                data = list(datasets.values())[0]
            else:
                data = datasets
    return data 


def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
    """
    Optimized function using R-tree for fast bounding box lookup.
    """
    length = coordinates_h5.shape[0]
    label = np.zeros(length, dtype=int)  # Efficient integer array for labels

    # Build R-tree index for fast spatial searching
    rtree_index = index.Index()

    for i, box in enumerate(coordinates_h5):  # Ensure h5_data is a NumPy array
        ymax, xmax, ymin, xmin = box  
        rtree_index.insert(i, (xmin, ymin, xmax, ymax))  

    # Iterate efficiently over DataFrame rows
    for row in tqdm(coordinates_xml.itertuples(index=False), desc="Checking index:", total=len(coordinates_xml), ncols=100):
        x, y = row.X, row.Y
        possible_matches = list(rtree_index.intersection((x, y, x, y)))  
        print(possible_matches)
        for box_index in possible_matches:
            if check_coor(x, y, coordinates_h5[box_index]): 
                label[box_index] = 1  # Mark as tumor

    return label 

def check_coor(x, y, box):
    """
    Checks if (x, y) is inside the given bounding box.
    """
    ymax, xmax, ymin, xmin = box  
    return xmin <= x <= xmax and ymin <= y <= ymax  # True if inside the bounding box
  

# def read_all_xml_file_base_tumor(file_h5_name):
#     xml_path = None
#     for path in list_xml_file:
#         if path.split("/")[-1] == file_h5_name:
#             xml_path = path
#     if not xml_path:
#         return pd.DataFrame()  # incase normal dont have file
#     coordinates_xml = return_df_xml(xml_path)
#     return coordinates_xml




# def check_coor(x, y, box):
#     ymax, xmax, ymin, xmin = box
#     if xmin <= x <= xmax and ymin <= y <= ymax:
#         return True  # The point is inside the bounding box
#     else:
#         return False  # The point is outside the bounding box


# def check_list_coor(x, y, list_coor, list_result):
#     for index, coor in enumerate(list_coor):  # Use enumerate to get the index directly
#         bool_check = check_coor(x, y, coor)
#         if bool_check is True:
#             list_result[index] = 1  # Set the corresponding label to 1
#             # print("tumor")
#     return list_result


# def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
#     length = coordinates_h5.shape[0]
#     label = np.zeros(length)  # Initialize label as a 1D array of zeros

#     for index, row in tqdm(coordinates_xml.iterrows(), desc="Checking index:", ncols=100):
#         label = check_list_coor(row["X"], row["Y"], coordinates_h5, label)
#         # print("Already check index: ", index)
#     return label


# original code 
# def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
#     length = coordinates_h5.shape[0]
#     label = np.zeros(length)  # Initialize label as a 1D array of zeros

#     for row in tqdm(coordinates_xml.itertuples(index=False), desc="Checking index:", total=len(coordinates_xml), ncols=100):
#         label = check_list_coor(row.X, row.Y, coordinates_h5, label)
#     return label

# import numpy as np
# from scipy.spatial import cKDTree
# from tqdm import tqdm


# def check_list_coor(x, y, tree, list_result, threshold=0):
#     """
#     Efficiently checks if (x, y) is within any bounding box using KDTree.
#     """
#     # Query the nearest bounding box
#     distances, indices = tree.query(np.array([[x, y]]), k=1)  # Nearest neighbor search

#     for dist, idx in zip(distances, indices):
#         if dist <= threshold:  # Use threshold if needed for slight tolerance
#             list_result[idx] = 1  # Mark as tumor

#     return list_result

# def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
#     """
#     Optimized function using KDTree for fast bounding box lookup.
#     """
#     length = coordinates_h5.shape[0]
#     label = np.zeros(length, dtype=int)  # Efficient integer array for labels

#     # Extract center points of bounding boxes (approximation)
#     box_centers = np.column_stack(((coordinates_h5[:, 1] + coordinates_h5[:, 3]) / 2,  # X center
#                                    (coordinates_h5[:, 0] + coordinates_h5[:, 2]) / 2)) # Y center

#     # Build KDTree for fast spatial queries
#     tree = cKDTree(box_centers)

#     # Iterate efficiently over DataFrame rows
#     for row in tqdm(coordinates_xml.itertuples(index=False), desc="Checking index:", total=len(coordinates_xml), ncols=100):
#         label = check_list_coor(row.X, row.Y, tree, label)

#     return label

