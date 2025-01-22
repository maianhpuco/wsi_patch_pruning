import os
import argparse 
import yaml  
import random

available = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032'] 
added = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']

def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
 
 
def main(args):
    # List all items in slide_path with .tif files
    slide_items = [i.split(".")[0] for i in os.listdir(args.slide_path) if i.endswith('tif')]
    print("Items in slide path:")
    for item in slide_items:
        print(item)

    # List all items in json_path with .json files
    json_items = [i.split('.')[0] for i in os.listdir(args.json_path) if i.endswith('json')]
    print("Items in json path:")
    for item in json_items:
        print(item)
        
    items_not_in_json = [item for item in slide_items if item not in json_items]
 
    # Find items in slide_items that are not in json_items
    items_not_in_json = [item for item in items_not_in_json if not item.startswith("test_")]

    sampled_items = random.sample(items_not_in_json, min(30, len(items_not_in_json)))

    print("Randomly sampled items not in json path (excluding 'test_' items):")
    print(sampled_items)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
          
    args.slide_path = config.get('SLIDE_PATH')
    args.json_path = config.get('JSON_PATH')    
    
    main(args)    
    
    
    
    # slide_items = [i.split(".")[0] for i in os.listdir(args.slide_path) if i.endswith('tif')]
    # json_items = [i.split('.')[0] for i in os.listdir(args.json_path) if i.endswith('json')]
    # items_not_in_json = [item for item in slide_items if item not in json_items] 
    # sampled_items = items_not_in_json  
    # items_not_in_json = [item for item in items_not_in_json if not item.startswith("test_")]
      