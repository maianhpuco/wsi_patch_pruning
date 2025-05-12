
### Preprocessing into patches: 
if we have to preprocessing: should be similar to this (but need to replace data with Camelyon16):  https://github.com/maianhpuco/wsi-data 


### Train mil classifier: 
- input: 
    FEATURES_H5_PATH: path/to/h5/features
    FEATURE_MEAN_STD_PATH: path/to/save/mean_std.h5
    SPLIT_PATH: path/to/split.csv
    CHECKPOINT_PATH: path/to/save/checkpoints
    PATCH_PATH: path/to/patches
    SLIDE_PATH: path/to/slides
    JSON_PATH: path/to/json
    SPIXEL_PATH: path/to/superpixels
    batch_size: 64
    feature_extraction_model: 'some_model'

- output: /path/to/checkpoints/mil_checkpoint_exp002.pth
- run: 
 
```
make classifing 
```

### PREDICT: predict on test set and return the result 
- input: 
    FEATURES_H5_PATH	Directory containing .h5 feature files per WSI
    FEATURE_MEAN_STD_PATH	Path to store or load mean/std stats
    SPLIT_PATH	CSV file specifying which WSIs belong to test/val
    CHECKPOINT_PATH	Folder containing model checkpoints
    PRED_PATH	Path to save test predictions
    batch_size	Batch size for evaluation
    Others (optional)	Slide path, patch path, etc.
- output: <PRED_PATH> 

```
make predicting 
```

### Get IG from classification

```
make group1 
make group2 
make group3 
``` 


### TABLE
in this folder: 
https://drive.google.com/drive/folders/1Y9Rx7ibHpCXtR30aSRHXvw26vvM3uZE7 

data is in this (h5):
https://drive.google.com/drive/folders/1tgff35Qx2CpvW1YUfPoWtL820tdVbZ4X?fbclid=IwAR0gqG32lev8DoP8F4L68ou8uRLj_fqIsKzJCEgdYe-anZ5NPCaPvwBY54s