import os
import sys
import glob
import time
import yaml
import h5py
import argparse
import numpy as np
import torch
import timm
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR)  


from data.merge_dataset import SlidePatchesDataset


 
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    print(f"Running {'dry run' if args.dry_run else 'full data processing'}")

    model = timm.create_model(args.feature_extraction_model, pretrained=True)
    model.to(args.device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wsi_paths = glob.glob(os.path.join(args.slide_path, '*.tif'))
    wsi_paths = [p for p in wsi_paths if os.path.basename(p).split(".")[0] in args.example_list]

    for count, wsi_path in enumerate(wsi_paths, start=1):
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        print(f">------ Processing {count}/{len(wsi_paths)}: {slide_basename}")

        slide_patch_dataset = SlidePatchesDataset(
            patch_dir=os.path.join(args.patch_path, slide_basename),
            transform=transform
        )
        dataloader = DataLoader(slide_patch_dataset, batch_size=args.batch_size, shuffle=True)

        slide_features_list = []
        patch_indices_list = []
        coordinates_list = []
        spixel_idx_list = []

        start_time = time.time()

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Extracting features from {slide_basename}"):
                batch_image = batch['image'].to(args.device)
                batch_patch_info = batch['patch_info']

                # Extract feature vectors
                batch_features = model.forward_features(batch_image)[:, 0, :].cpu()  # Move to CPU immediately

                # Collect patch indices, coordinates, and superpixel indices
                batch_idxes = [batch_patch_info['patch_idx'][i].item() for i in range(len(batch_patch_info['patch_idx']))]
                batch_coordinates = [[
                    batch_patch_info['xmin'][i].item(),
                    batch_patch_info['xmax'][i].item(),
                    batch_patch_info['ymin'][i].item(),
                    batch_patch_info['ymax'][i].item()
                ] for i in range(len(batch_patch_info['xmin']))]
                batch_spixel_idx = [batch_patch_info['spixel_idx'][i].item() for i in range(len(batch_patch_info['spixel_idx']))]

                # Append extracted data
                slide_features_list.append(batch_features)  # Already on CPU
                patch_indices_list.extend(batch_idxes)
                coordinates_list.extend(batch_coordinates)
                spixel_idx_list.extend(batch_spixel_idx)

                # Clear GPU memory
                del batch_image, batch_features
                torch.cuda.empty_cache()

        # Stack all features to a single tensor (on CPU)
        slide_features = torch.cat(slide_features_list, dim=0)
        patch_indices = np.array(patch_indices_list)
        coordinates = np.array(coordinates_list)
        spixel_idx = np.array(spixel_idx_list)

        # Determine label (0 = normal, 1 = tumor)
        label = 0 if slide_basename.startswith("normal") else 1

        # Save extracted features to HDF5 file
        output_file = os.path.join(args.features_h5_path, f"{slide_basename}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=slide_features.numpy())  # Save to disk
            f.create_dataset('patch_indices', data=patch_indices)
            f.create_dataset('label', data=np.array([label]))
            f.create_dataset('coordinates', data=coordinates)
            f.create_dataset('spixel_idx', data=spixel_idx)

        print(f"Saved features for {slide_basename} to {output_file}")
        print(f"Processing time: {((time.time() - start_time) / 60):.2f} mins")
        print(f"Feature shape: {slide_features.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()

    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        args.slide_path = config.get('SLIDE_PATH')
        args.patch_path = config.get('PATCH_PATH')
        args.features_h5_path = config.get("FEATURES_H5_PATH")
        os.makedirs(args.features_h5_path, exist_ok=True)

        args.batch_size = config.get('batch_size', 4)  # Reduce batch size if needed
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args.example_list = ['tumor_026']  # Modify for testing
    args.example_list = ['normal_114', 'tumor_026', 'tumor_009', 'tumor_024', 'tumor_015', 'normal_076','normal_070', 'normal_066', 'normal_053', 'normal_104','normal_112']   

    main(args)
 