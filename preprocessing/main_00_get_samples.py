import os
import argparse 
import yaml  
import random

available = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032'] 

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

    # Find items in slide_items that are not in json_items
    items_not_in_json = [item for item in slide_items if item not in json_items]
    
    # Randomly sample 20 items from those not in json
    sampled_items = random.sample(items_not_in_json, min(20, len(items_not_in_json)))

    print("Randomly sampled items not in json path:")
    for item in sampled_items:
        print(item) 
        

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