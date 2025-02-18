import numpy as np


def calculate_tp(mask, predict):
    tp = np.sum((mask == 1) & (predict == 1))
    return tp


def calculate_fp(mask, predict):
    fp = np.sum((mask == 0) & (predict == 1))
    return fp


def calculate_tn(mask, predict):
    tn = np.sum((mask == 0) & (predict == 0))
    return tn


def calculate_fn(mask, predict):
    fn = np.sum((mask == 1) & (predict == 0))
    return fn


def calculate_dice_score(mask, predict):
    tp = calculate_tp(mask, predict)
    fp = calculate_fp(mask, predict)
    fn = calculate_fn(mask, predict)

    # Avoid division by zero
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 1.0  # Perfect prediction
    dice = (2 * tp) / denominator
    return dice


def calculate_iou_score(mask, predict):
    tp = calculate_tp(mask, predict)
    fp = calculate_fp(mask, predict)
    fn = calculate_fn(mask, predict)

    # Avoid division by zero
    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0  # Perfect prediction
    iou = tp / denominator
    return iou
