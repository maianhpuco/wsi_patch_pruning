path_h5 = (
    "/Users/nam.le/Desktop/research/camil_pytorch/data/camelyon16_feature/h5_files"
)

import os
from utils_coreset import list_files, read_h5_data, check_data, store_json
from method.uniform import sample_features_indices as uniform_method
from method.kcenter import k_center_greedy as k_center
from method.imp_scr import sample_important_indices as imp_scr
from method.gradient import gradient_complete_function as gradient
import time

method_dict = {"uniform": uniform_method, "kcenter": k_center, "imp_scr": imp_scr}


def main():
    list_path_h5 = list_files(path_h5)
    # print(len(list_path_h5))
    # print(list_path_h5)
    list_path_h5 = [
        "/Users/nam.le/Desktop/research/camil_pytorch/data/camelyon16_feature/h5_files/tumor_048.h5"
    ]
    data = {"time": 0}
    start_time = time.time()
    for data_path in list_path_h5:
        if data_path[-1] != "5":
            continue
        img_path = os.path.basename(data_path)
        data_sample = read_h5_data(data_path)
        check_data(data_sample, data_path)
        break
        # change_method
        # list_store = uniform_method(data_sample["features"])
        # data_sample["features"] = data_sample["features"][:50]
        # list_store = gradient(data_sample["features"])
        data[img_path] = list_store if type(list_store) is list else list_store.tolist()
    end_time = time.time()
    elapsed_time = end_time - start_time
    data["time"] = elapsed_time
    store_json(data, "k_center")


if __name__ == "__main__":
    main()
