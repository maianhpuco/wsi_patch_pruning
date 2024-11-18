# camil in pytorch


Run 
activate conda/pip env and then:  

```export PROJECT_DIR=$(pwd)``` 
before start env 


down load this file
```!gdown 1CS7I0yrTSNLbFk_CzqLrh5TKesZo3uXm ``` 
then unzip them into ```data/camelyon16_feature/h5_files```

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