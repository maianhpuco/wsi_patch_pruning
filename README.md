
# TEST BED FOR PATCH MERGING: 

## About test bed:
#### Todo before running main file for test bed: 
1. activate conda/pip env and then:  
```export PROJECT_DIR=$(pwd)```  
2. Fix your path in testbed_config/{yourname_expnumber.yaml}



#### Main file:
    ```python main_testbed_slide.py``` -> loop through all slide and return the patch (image) in the whole slide 
    
    ```python main_testbed_slide.py``` -> loop through each slide, then loop through all superixel and return the patch (image) in the whole slide  



# Camil Training: 
----- Below config is to run Camil 
before start env 
```export DATA_DIR=/project/hnguyen2/mvu9/camelyon16_features_data/h5_files ```

down load this file
```!gdown 1CS7I0yrTSNLbFk_CzqLrh5TKesZo3uXm ``` 
then unzip them into ```data/camelyon16_feature/h5_files```

- to running the training: 
```python train.py```
- to dry run (testing the code with few sample), run:
```python train.py --dry_run True```

``` 
.
├── README.md
├── check_cuda.py
├── data
│   ├── camelyon16_dataset.py
│   ├── camelyon16_features
│   │   └── h5_files
│   ├── camelyon_csv_splits
│   │   ├── splits_0.csv
│   │   ├── splits_1.csv
│   │   ├── splits_2.csv
│   │   ├── splits_3.csv
│   │   └── splits_4.csv
│   ├── label_files
│   │   ├── camelyon_17.csv
│   │   ├── camelyon_data.csv
│   │   └── tcga_data.csv
│   ├── logs
│   └── weights
├── feature_extractor
├── requirements.txt
├── scripts
├── src
│   ├── __init__.py
│   ├── camil.py
│   ├── custom_layers.py
│   ├── nystromformer.py
├── train.py
└── utils
    ├── __init__.py
    ├── eval.py
    ├── helper.py
    └── utils.py 

```
Experiment 01: 
- use pretrained embedding
- learning rate: 1e-05
- epochs: 30 
 
 ![image](https://github.com/user-attachments/assets/91b3114e-57a1-4cb4-9e81-2b3dda59f5a8)


