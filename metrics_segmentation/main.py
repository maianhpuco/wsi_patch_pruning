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


def main():
    # 0 is back ground, 1 is tumor
    mask = np.random.randint(0, 2, size=(30000, 1))
    predict = np.random.randint(0, 2, size=(30000, 1))

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
