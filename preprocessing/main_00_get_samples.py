import os
available = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032'] 

def main(args):
    # List all items in slide_path
    slide_items = os.listdir(args.slide_path)
    print("Items in slide path:")
    for item in slide_items:
        print(item)

    # List all items in json_path
    json_items = os.listdir(args.json_path)
    print("Items in json path:")
    for item in json_items:
        print(item) 


if __name__ ==__'__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    args.slide_path = config.get('SLIDE_PATH')
    args.json_path = config.get('JSON_PATH')    
    
    main(args)     