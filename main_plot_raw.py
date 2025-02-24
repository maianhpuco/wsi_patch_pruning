import matplotlib.pyplot as plt
import os
import argparse
import h5py
import openslide
import torch
from utils.utils import load_config
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
) 

def load_h5_file(h5_path):
    """
    Load data from an H5 file.
    
    Args:
        h5_path (str): Path to the H5 file
    
    Returns:
        dict: Dictionary containing the loaded data
    """
    dict_data = {}
    with h5py.File(h5_path, "r") as f:
        # Load all datasets from the h5 file
        for key in f.keys():
            dict_data[key] = f[key][:]
            
    return dict_data

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='ma_exp002')
    parser.add_argument('--show_plot', type=int, default=0)
    parser.add_argument('--wsi_name', default='tumor_026')
    parser.add_argument('--dry_run', type=int, default=0)
    
    args = parser.parse_args()

    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
        args.checkpoints_dir = config.get("CHECKPOINT_PATH")
        if args.dry_run==1:
            args.attribution_scores_folder = config.get("SCORE_FOLDER_DRYRUN") 
            args.plot_path = config.get("PLOT_PATH_DRYRUN")    
        else: 
            args.attribution_scores_folder = config.get("SCORE_FOLDER")    
            args.plot_path = config.get("PLOT_PATH")
            
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # plot raw slide
    slide = openslide.open_slide(os.path.join(args.slide_path, f'{args.wsi_name}.tif'))
    (
        downsample_factor,
        new_width,
        new_height,
        original_width,
        original_height
    ) = rescaling_stat_for_segmentation(slide, downsampling_size=1096)

    image_numpy = downscaling(slide, new_width, new_height)

    figsize=(20, 20)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_numpy)
    ax.axis('off')

    plt.savefig(os.path.join(args.plot_path, f'{args.wsi_name}_raw.png'), bbox_inches='tight', dpi=100)
    if args.show_plot:
        plt.show()
    
if __name__ == "__main__":
    __main__()