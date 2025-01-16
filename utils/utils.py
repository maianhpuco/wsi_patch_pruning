import os
import sys 


def get_region_original_size(slide, xywh_abs_bbox):
    xmin_original, ymin_original, width_original, height_original = xywh_abs_bbox
    region = slide.read_region(
        (xmin_original, ymin_original),  # Top-left corner (x, y)
        0,  # Level 0
        (width_original, height_original)  # Width and height
    )
    return region.convert('RGB')