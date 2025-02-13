import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import openslide
from matplotlib import cm
import math
import numpy as np
from torchvision import transforms
from skimage.transform import resize
import os 
from tqdm import tqdm  

def min_max_scale(array):
    min_val = np.min(array)
    max_val = np.max(array)

    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(array)

    return (array - min_val) / (max_val - min_val) 


def replace_outliers_with_bounds(array):
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace values below lower bound with lower bound and above upper bound with upper bound
    array = np.where(array < lower_bound, lower_bound, array)
    array = np.where(array > upper_bound, upper_bound, array)

    return array  

def rescaling_stat_for_segmentation(obj, downsampling_size=1024):
    """
    Rescale the image to a new size and return the downsampling factor.
    """
    if hasattr(obj, 'shape'):
        original_width, original_height = obj.shape[:2]
    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        original_width, original_height = obj.size
    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        original_width, original_height = obj.dimensions
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

    if original_width > original_height:
        downsample_factor = int(downsampling_size * 100000 / original_width) / 100000
    else:
        downsample_factor = int(downsampling_size * 100000 / original_height) / 100000

    new_width = int(original_width * downsample_factor)
    new_height = int(original_height * downsample_factor)

    return downsample_factor, new_width, new_height, original_width, original_height

def downscaling(obj, new_width, new_height):
    """
    Downscale the given object (image or slide) to the specified size.
    """
    if isinstance(obj, np.ndarray):  # If it's a NumPy array
        # Resize using scikit-image (resize scales and interpolates)
        image_numpy = resize(obj, (new_height, new_width), anti_aliasing=True)
        image_numpy = (image_numpy * 255).astype(np.uint8)

    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        obj = obj.resize((new_width, new_height))
        image_numpy = np.array(obj)

    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        thumbnail = obj.get_thumbnail((new_width, new_height))
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a tensor (C, H, W)
        ])
        image_tensor = transform(thumbnail)
        image_numpy = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C) numpy format
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

    return image_numpy


def get_region_original_size(slide, xywh_abs_bbox):
    xmin_original, ymin_original, width_original, height_original = xywh_abs_bbox
    region = slide.read_region(
        (xmin_original, ymin_original),  # Top-left corner (x, y)
        0,  # Level 0
        (width_original, height_original)  # Width and height
    )
    return region.convert('RGB')


def plot_image_with_bboxes(basename, SLIDE_PATH, coordinates, scores, figsize=(20, 20)):

    slide_path = os.path.join(SLIDE_PATH, f'{basename}.tif')

    # Open slide image using OpenSlide
    slide = openslide.open_slide(slide_path)

    downsample_factor, new_width, new_height, original_width, original_height = rescaling_stat_for_segmentation(slide, downsampling_size=1096)

    # Downscale the slide to the new dimensions
    image_numpy = downscaling(slide, new_width, new_height)
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Normalize the scores to the range [0, 1] for color mapping
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # Set the figure size for the 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Define the colormap (from blue to red)
    cmap = cm.get_cmap('coolwarm')  # 'coolwarm' goes from blue to red, but you can use any colormap
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))  # Same normalization for both subplots

    # First subplot: Original image with bounding boxes
    ax = axes[0]

    # Display the original image
    ax.imshow(image_numpy)

    # Hide axes for better visualization
    ax.axis('off')

    # Add color bar to the first subplot
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label('Score Value', rotation=270, labelpad=15)

    # Second subplot: Heatmap on white background with semi-transparent bounding boxes
    ax2 = axes[1]

    # Create a white background (just a blank canvas)
    white_background = np.ones((new_height, new_width, 3))  # White image of the same size as downscaled image
    ax2.imshow(white_background)

    # Iterate through the bounding boxes and draw the semi-transparent boxes based on the normalized score
    for i, bbox in enumerate(coordinates):
        ymax, xmax, ymin, xmin = bbox.astype('int')

        # Scale the bounding box coordinates
        scaled_xmin = xmin * scale_x
        scaled_xmax = xmax * scale_x
        scaled_ymin = ymin * scale_y
        scaled_ymax = ymax * scale_y

        # Get the normalized score for the bounding box
        score = norm_scores[i]

        # Get the color based on the normalized score
        color = cmap(score)  # This gives an RGBA value based on the normalized score

        # Create a semi-transparent rectangle for the bounding box with the corresponding color
        rect = patches.Rectangle(
            (scaled_xmin, scaled_ymin),
            scaled_xmax - scaled_xmin,
            scaled_ymax - scaled_ymin,
            linewidth=0.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.5  # Set alpha for transparency
        )
        ax2.add_patch(rect)

    # Hide axes for better visualization in the second subplot
    ax2.axis('off')

    # Add color bar for the heatmap (same color bar as the first subplot)
    fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax2, label='Score Value')

    # Show the plot with both subplots
    plt.show()


def plot_heatmap_with_bboxes(
    scale_x,scale_y, 
    new_height, new_width, 
    coordinates, 
    scores, 
    figsize=(10, 10), 
    name="", 
    save_path=None):
    
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # Define colormap
    # cmap = cm.get_cmap('coolwarm')
    cmap=cm.get_cmap('jet')
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    # Create a figure for the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Create a white background
    white_background = np.ones((new_height, new_width, 3))
    ax.imshow(white_background)

    # Iterate through bounding boxes and draw semi-transparent colored rectangles
    for i, bbox in tqdm(enumerate(coordinates), total=len(coordinates), desc="Plotting Bounding Boxes"):
    # for i, bbox in enumerate(coordinates):
        ymax, xmax, ymin, xmin = bbox.astype('int')

        # Scale bounding box coordinates
        scaled_xmin = xmin * scale_x
        scaled_xmax = xmax * scale_x
        scaled_ymin = ymin * scale_y
        scaled_ymax = ymax * scale_y

        # Get the color based on the normalized score
        color = cmap(norm_scores[i])

        # Create a semi-transparent bounding box
        rect = patches.Rectangle(
            (scaled_xmin, scaled_ymin),
            scaled_xmax - scaled_xmin,
            scaled_ymax - scaled_ymin,
            linewidth=0.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.5  # Transparency
        )
        ax.add_patch(rect)

    # Hide axes for better visualization
    ax.axis('off')

    # Add color bar
    fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='Score Value')


    plt.title(name, fontsize=10, fontweight='bold') 
    # Show the heatmap
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Ensure save_dir is not an empty string (for root files)
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"Saved heatmap to {save_path}")  
 
    plt.show()

