import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from utils.utils import load_config
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
) 

# List of image paths and titles
image_paths = [
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/tumor_026_raw.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/tumor_026_ground_truth.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/integrated_gradient/tumor_026.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/integrated_gradient/tumor_026.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/integrated_gradient/tumor_031.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/integrated_gradient/tumor_032.png",
    "/Users/mfv-computer-0261/Desktop/tmp/plotting_dryrun/vanilla_gradient/tumor_026.png"
]
titles = ["Raw", "Ground Truth", "Gradient", "Vanilla IG", "Vanilla IG", "Guided IG", "Ours"]

# Load the first image to set the target size for all images
first_image = Image.open(image_paths[0])
target_size = first_image.size  # (width, height)

num_images = len(image_paths)
fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 4))

for i, (ax, img_path, title) in enumerate(zip(axes, image_paths, titles)):
    img = Image.open(img_path).resize(target_size, Image.LANCZOS)
    img_with_border = ImageOps.expand(img, border=1, fill="black")

    # Example: if you highlight the last column
    if i == num_images - 1:
        img_with_border = ImageOps.expand(img, border=7, fill="red")
        # Convert to RGBA for blending
        img_rgba = img_with_border.convert("RGBA")
        # Create an overlay of color #e3c6c6 with 30% opacity
        overlay = Image.new("RGBA", img_rgba.size, (227, 198, 198, int(255 * 0.3)))

    ax.imshow(img_with_border)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# Remove all spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

plt.show()