import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.patches as patches 
import numpy as np

images_arr = [
    [
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/raw/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/ground_truth/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_gradient/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/expected_gradient/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_decision_gradient/test_021.png',
    
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/vanilla_gradient/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/square_integrated_gradient/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/optim_square_integrated_gradient/test_021.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/contrastive_gradient/test_021.png',
    ],
    [
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/raw/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/ground_truth/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/expected_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_decision_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/vanilla_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/square_integrated_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/optim_square_integrated_gradient/test_040.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/contrastive_gradient/test_040.png',
    ],
    [
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/raw/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/ground_truth/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/expected_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/integrated_decision_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/vanilla_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/square_integrated_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/optim_square_integrated_gradient/test_068.png',
    '/Users/mfv-computer-0261/Desktop/tmp/plotting/contrastive_gradient/test_068.png',
    ],
]

titles = ["Original", "GT", "IG", "EG", "IDG", "VG", "S-IG", "OS-IG", "CG"]


fig, ax = plt.subplots(3, 9, figsize=(8, 6))

# Add titles only to the first row
for j, title in enumerate(titles):
    ax[0, j].set_title(title, fontsize=8)



for i, images in enumerate(images_arr):
    for j, image_path in enumerate(images):
        img = Image.open(image_path)
        img_border = ImageOps.expand(img, border=1, fill="black")
        im = ax[i, j].imshow(img_border, alpha=1.0)  # Store the imshow object
        ax[i, j].axis('off')

# Add background for last column


color = ['r', 'b', 'g'][np.random.randint(3)]
bbox = ax[0, 0].get_window_extent()
fig.patches.extend([plt.Rectangle((1307, 150),
                                    bbox.width * 1.2, bbox.height * 3.5,
                                    fill=True, color=color, alpha=0.2, zorder=-100,
                                    transform=None, figure=fig)])




# Add colorbar at the bottom
plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.15)  # Make space for colorbar
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
cbar.ax.tick_params(labelsize=8)  # Adjust tick label size if needed

plt.show()
