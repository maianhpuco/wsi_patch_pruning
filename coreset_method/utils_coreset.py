import os
import h5py
import json


def list_files(folder_path):
    file_names = []

    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path):
            file_names.append(os.path.join(folder_path, item))

    return file_names


def read_h5_data(file_path, dataset_name=None):
    data = None
    with h5py.File(file_path, "r") as file:
        if dataset_name is not None:
            if dataset_name in file:
                dataset = file[dataset_name]
                data = dataset[()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
        else:
            datasets = {}

            def visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets[name] = node[()]

            file.visititems(visitor)

            if len(datasets) == 1:
                data = list(datasets.values())[0]
            else:
                data = datasets
    return data


def check_data(data):
    for key, value in data.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: does not have shape attribute")


def store_json(data, method):
    path = f"output/{method}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
