import numpy as np

np.random.seed(0)
from method import (
    calculate_dice_score,
    calculate_iou_score,
    calculate_fp,
    calculate_fn,
    calculate_tn,
    calculate_tp,
)
from metrics_segmentation.utils_metrics_nam import (
    read_all_xml_file_base_tumor,
    check_xy_in_coordinates,
    read_h5_data,
)


def main():

    # Assume that have path of h5 file
    path = ""
    # path = "/Users/nam.le/Desktop/research/camil_pytorch/data/camelyon16_feature/h5_files/tumor_048.h5"
    h5_name = path.split("/")[-1].replace("h5", "xml")
    
    df_xml = read_all_xml_file_base_tumor(h5_name)
    print(df_xml, type(df_xml))
    h5_data = read_h5_data(path)
    mask = check_xy_in_coordinates(df_xml, h5_data["coordinates"])
    
    # 0 is back ground, 1 is tumor
    predict = np.random.randint(0, 2, size=(h5_data["coordinates"].shape[0], 1))

    tp = calculate_tp(mask, predict)
    fp = calculate_fp(mask, predict)
    tn = calculate_tn(mask, predict)
    fn = calculate_fn(mask, predict)
    dice = calculate_dice_score(mask, predict)
    iou = calculate_iou_score(mask, predict)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")


if __name__ == "__main__":
    main()
