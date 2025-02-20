import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from rtree import index  # R-tree for fast spatial lookup
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # Ensures tqdm updates correctly with joblib 
import os 
import time 
import multiprocessing 
PATCH_SIZE = 224  # Define patch size (downscaling factor)

def parse_xml(file_path):
    """ Parse XML file and return root element. """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def downscale_coordinates(contour, scale_factor):
    """ Downscale contour coordinates for faster processing. """
    return [(x / scale_factor, y / scale_factor) for x, y in contour]

def upscale_coordinates(points, scale_factor):
    """ Upscale points back to original size. """
    return [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]

# from joblib import Parallel, delayed
# from tqdm import tqdm
# import os
# import numpy as np
# import pandas as pd
# from shapely.geometry import Polygon, Point
# from tqdm_joblib import tqdm_joblib  # Ensures tqdm updates correctly with joblib

def extract_coordinates_parallel(file_path, save_dir, max_cpus=8):
    """
    Parallelized extraction of (X, Y) coordinates inside a contour with tqdm progress bar.
    """
    basename = os.path.basename(file_path).split(".")[0]
    save_path = os.path.join(save_dir, f'{basename}.csv') 
    
    # Parse XML
    root = parse_xml(file_path)
    if root is None:
        return None  

    contour = [
        (float(coord.attrib["X"]), float(coord.attrib["Y"]))
        for coord in root.findall(".//Coordinate")
    ]

    if not contour:
        return None  

    # Downscale & create polygon
    downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)
    polygon = Polygon(downscaled_contour)

    min_x, min_y, max_x, max_y = polygon.bounds
    x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
    y_patches = np.arange(np.floor(min_y), np.ceil(max_y))

    def process_patch(x, y):
        return (x, y) if polygon.contains(Point(x, y)) else None

    inside_points = []

    # Use tqdm with joblib (ensuring correct updates)
    with tqdm_joblib(tqdm(desc="Processing Patches", total=len(x_patches) * len(y_patches), ncols=100)):
        inside_points = Parallel(n_jobs=max_cpus)(
            delayed(process_patch)(x, y) for x in x_patches for y in y_patches
        )

    inside_points = [p for p in inside_points if p]  # Remove None values
    original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

    result_df = pd.DataFrame({"File": basename, 
                              "X": [p[0] for p in original_size_points], 
                              "Y": [p[1] for p in original_size_points]})

    result_df.to_csv(save_path, index=False)  # Fix `index_col=0` (not needed)
    
    return result_df
 
 
# import time
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import os
# import numpy as np
# import pandas as pd
# from shapely.geometry import Polygon, Point

import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

def extract_coordinates_parallel(file_path, save_dir, max_cpus=8, batch_size=500):
    """
    Parallelized batch extraction of (X, Y) coordinates inside a contour with tqdm.
    """
    basename = os.path.basename(file_path).split(".")[0]
    save_path = os.path.join(save_dir, f'{basename}.csv') 
    
    # Parse XML
    root = parse_xml(file_path)
    if root is None:
        print(f"Skipping {basename} (invalid XML)")
        return None  

    contour = [
        (float(coord.attrib["X"]), float(coord.attrib["Y"]))
        for coord in root.findall(".//Coordinate")
    ]

    if not contour:
        print(f"Skipping {basename} (no contour found)")
        return None  

    # Downscale & create polygon
    downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)
    polygon = Polygon(downscaled_contour)

    # ✅ **Use np.meshgrid() to properly create a grid**
    min_x, min_y, max_x, max_y = polygon.bounds
    x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
    y_patches = np.arange(np.floor(min_y), np.ceil(max_y))
    x_grid, y_grid = np.meshgrid(x_patches, y_patches)  # Correct grid formation
    xy_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])  # Flatten

    total_patches = len(xy_points)

    print(f"Total CPUs available: {multiprocessing.cpu_count()}")
    print(f"Using {max_cpus} CPUs for parallel processing")
    print(f"Total patches to process: {total_patches}")

    # ✅ **Batch processing function**
    def process_patch_batch(batch):
        """Check which patches are inside the polygon in a batch."""
        return [(x, y) for x, y in batch if polygon.contains(Point(x, y))]

    # Create batches of patches
    point_batches = [xy_points[i:i+batch_size] for i in range(0, total_patches, batch_size)]

    start = time.time()
    
    # ✅ **Use tqdm with correct updates in parallel loop**
    inside_points_batches = []
    with tqdm(total=len(point_batches), desc=f"Processing {basename}", ncols=100) as pbar:
        for batch in Parallel(n_jobs=max_cpus)(delayed(process_patch_batch)(b) for b in point_batches):
            inside_points_batches.append(batch)
            pbar.update(1)  # Update after each batch

    # Flatten the list correctly
    inside_points = [p for batch in inside_points_batches for p in batch]

    # ✅ **Fix coordinate scaling issue**
    original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

    result_df = pd.DataFrame({"File": basename, 
                              "X": [p[0] for p in original_size_points], 
                              "Y": [p[1] for p in original_size_points]})

    result_df.to_csv(save_path, index=False)  
    print(f"Completed {basename} in {(time.time()-start)/60:.2f} min")
    
    return result_df


# def extract_coordinates_parallel(file_path, save_dir, max_cpus=8, batch_size=500):
#     """
#     Parallelized extraction of (X, Y) coordinates inside a contour with tqdm progress bar.
#     """
#     basename = os.path.basename(file_path).split(".")[0]
#     save_path = os.path.join(save_dir, f'{basename}.csv') 
    
#     # Parse XML
#     root = parse_xml(file_path)
#     if root is None:
#         print(f"Skipping {basename} (invalid XML)")
#         return None  

#     contour = [
#         (float(coord.attrib["X"]), float(coord.attrib["Y"]))
#         for coord in root.findall(".//Coordinate")
#     ]

#     if not contour:
#         print(f"Skipping {basename} (no contour found)")
#         return None  

#     # Downscale & create polygon
#     downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)
#     polygon = Polygon(downscaled_contour)

#     # ✅ **Use np.meshgrid() to properly create a grid**
#     min_x, min_y, max_x, max_y = polygon.bounds
#     x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
#     y_patches = np.arange(np.floor(min_y), np.ceil(max_y))
#     x_grid, y_grid = np.meshgrid(x_patches, y_patches)  # Correct grid formation
#     xy_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])  # Flatten

#     total_patches = len(xy_points)

#     print(f"Total CPUs available: {multiprocessing.cpu_count()}")
#     print(f"Using {max_cpus} CPUs for parallel processing")
#     print(f"Total patches to process: {total_patches}")

#     # ✅ **Fix: Ensure batch slicing is correct**
#     def process_patch_batch(batch):
#         """Check which patches are inside the polygon in a batch."""
#         return [(x, y) for x, y in batch if polygon.contains(Point(x, y))]

#     # Create **correctly sliced** batches
#     point_batches = [xy_points[i:i+batch_size] for i in range(0, total_patches, batch_size)]

#     start = time.time()
    
#     # ✅ **Fix tqdm inside parallel loop**
#     inside_points_batches = []
#     with tqdm(total=len(point_batches), desc=f"Processing {basename}", ncols=100) as pbar:
#         for batch in Parallel(n_jobs=max_cpus)(delayed(process_patch_batch)(b) for b in point_batches):
#             inside_points_batches.append(batch)
#             pbar.update(1)  # Update after each batch

#     # Flatten the list correctly
#     inside_points = [p for batch in inside_points_batches for p in batch]

#     # ✅ **Fix coordinate scaling issue**
#     original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

#     result_df = pd.DataFrame({"File": basename, 
#                               "X": [p[0] for p in original_size_points], 
#                               "Y": [p[1] for p in original_size_points]})

#     result_df.to_csv(save_path, index=False)  
#     print(f"Completed {basename} in {(time.time()-start)/60:.2f} min")
    
#     return result_df

 
# def extract_coordinates_parallel(file_path, save_dir, max_cpus=8):
#     """
#     Parallelized extraction of (X, Y) coordinates inside a contour with tqdm progress bar.
#     """
#     basename = os.path.basename(file_path).split(".")[0]
#     save_path = os.path.join(save_dir, f'{basename}.csv') 
    
#     # Parse XML
#     root = parse_xml(file_path)
#     if root is None:
#         return None  

#     contour = [
#         (float(coord.attrib["X"]), float(coord.attrib["Y"]))
#         for coord in root.findall(".//Coordinate")
#     ]

#     if not contour:
#         return None  

#     # Downscale & create polygon
#     downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)
#     polygon = Polygon(downscaled_contour)

#     min_x, min_y, max_x, max_y = polygon.bounds
#     x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
#     y_patches = np.arange(np.floor(min_y), np.ceil(max_y))

#     def process_patch(x, y):
#         return (x, y) if polygon.contains(Point(x, y)) else None

#     inside_points = []
#     start = time.time()
#     # Use tqdm with joblib (ensuring correct updates)
#     with tqdm_joblib(tqdm(desc="Processing Patches", total=len(x_patches) * len(y_patches), ncols=100)):
#         inside_points = Parallel(n_jobs=max_cpus)(
#             delayed(process_patch)(x, y) for x in x_patches for y in y_patches
#         )

#     inside_points = [p for p in inside_points if p]  # Remove None values
#     original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

#     result_df = pd.DataFrame({"File": basename, 
#                               "X": [p[0] for p in original_size_points], 
#                               "Y": [p[1] for p in original_size_points]})

#     result_df.to_csv(save_path, index=False)  # Fix `index_col=0` (not needed)
#     print(f"Complete process computing coordinate ground truth after: {(time.time()-start)/60.000}")
#     return result_df
 
 

    
# def extract_coordinates_parallel(file_path, save_dir):
#     """
#     Parallelized extraction of (X, Y) coordinates inside a contour with tqdm progress bar.
#     """
#     basename = os.path.basename(file_path).split(".")[0]
#     save_path = os.path.join(save_dir, f'{basename}.csv') 
    
#     # Parse XML
#     root = parse_xml(file_path)
#     if root is None:
#         return None  

#     contour = [
#         (float(coord.attrib["X"]), float(coord.attrib["Y"]))
#         for coord in root.findall(".//Coordinate")
#     ]

#     if not contour:
#         return None  

#     # Downscale & create polygon
#     downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)
#     polygon = Polygon(downscaled_contour)

#     min_x, min_y, max_x, max_y = polygon.bounds
#     x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
#     y_patches = np.arange(np.floor(min_y), np.ceil(max_y))

#     inside_points = []

#     # Use tqdm with joblib (manual update)
#     with tqdm(total=len(x_patches) * len(y_patches), desc="Processing Patches", ncols=100) as pbar:
#         def process_patch(x, y):
#             result = (x, y) if polygon.contains(Point(x, y)) else None
#             pbar.update(1)  # Update progress manually
#             return result

#         inside_points = Parallel(n_jobs=-1)(
#             delayed(process_patch)(x, y) for x in x_patches for y in y_patches
#         )

#     inside_points = [p for p in inside_points if p]  # Remove None values
#     original_size_points = upscale_coordinates(inside_points, PATCH_SIZE) 

#     result_df = pd.DataFrame({"File": basename, 
#                          "X": [p[0] for p in original_size_points], 
#                          "Y": [p[1] for p in original_size_points]})
#     result_df.to_csv(save_path, index_col=0)
#     return result_df 
 
 
def extract_coordinates(file_path):
    """
    Fast extraction of (X, Y) coordinates **inside** a contour using downscaling and R-tree.
    """
    root = parse_xml(file_path)
    if root is None:
        return None  # Skip if parsing failed

    print(f"Processing XML: {file_path}")

    # Extract contour points
    contour = []
    for coordinate in root.findall(".//Coordinate"):
        x = coordinate.attrib.get("X")
        y = coordinate.attrib.get("Y")
        if x and y:
            contour.append((float(x), float(y)))

    if not contour:
        return None  # No contour found

    # Downscale the contour for faster processing
    downscaled_contour = downscale_coordinates(contour, PATCH_SIZE)

    # Convert to a Shapely polygon (downscaled)
    polygon = Polygon(downscaled_contour)

    # Generate downscaled grid (treat each patch as a "pixel")
    min_x, min_y, max_x, max_y = polygon.bounds
    x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
    y_patches = np.arange(np.floor(min_y), np.ceil(max_y))

    inside_points = []

    # Use tqdm to track progress
    with tqdm(total=len(x_patches) * len(y_patches), desc="Processing Patches", ncols=100) as pbar:
        for x_start in x_patches:
            for y_start in y_patches:
                patch_point = Point(x_start, y_start)

                # Check if the patch (downscaled pixel) is inside
                if polygon.contains(patch_point):
                    inside_points.append((x_start, y_start))  # Save downscaled points
                
                pbar.update(1)

    # Upscale the points back to original resolution
    original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

    # Convert to DataFrame
    df_inside_points = pd.DataFrame({"File": file_path.split("/")[-1], "X": [p[0] for p in original_size_points], "Y": [p[1] for p in original_size_points]})
    
    return df_inside_points

def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
    """
    Optimized function using R-tree for fast bounding box lookup.
    """
    length = coordinates_h5.shape[0]
    label = np.zeros(length, dtype=int)  # Efficient integer array for labels

    # Build R-tree index for fast spatial searching
    rtree_index = index.Index()
    for i, box in enumerate(coordinates_h5):  
        ymax, xmax, ymin, xmin = box  
        rtree_index.insert(i, (xmin, ymin, xmax, ymax))  

    # Iterate efficiently over DataFrame rows
    for row in tqdm(coordinates_xml.itertuples(index=False), desc="Checking index:", total=len(coordinates_xml), ncols=100):
        x, y = row.X, row.Y
        possible_matches = list(rtree_index.intersection((x, y, x, y)))  

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

def check_xy_in_coordinates_fast(coordinates_xml, coordinates_h5):
    """
    Vectorized function using R-tree for fast lookup.
    """
    label = np.zeros(len(coordinates_h5), dtype=np.int8)  

    rtree_index = index.Index((i, (xmin, ymin, xmax, ymax), None) for i, (ymax, xmax, ymin, xmin) in enumerate(coordinates_h5))

    xy_pairs = np.column_stack((coordinates_xml["X"], coordinates_xml["Y"]))  # Convert to NumPy array for vectorized operations

    for i, (x, y) in enumerate(xy_pairs):
        possible_matches = list(rtree_index.intersection((x, y, x, y)))  
        
        if possible_matches:
            label[possible_matches] = 1  # Vectorized assignment

    return label
 