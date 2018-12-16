# Find Objects

import cv2
import numpy as np
import os



def find_objects(img, mask):
    """Find all objects and color them blue.
    """

    params.device += 1
    mask1 = np.copy(mask)
    ori_img = np.copy(img)
    # If the reference image is grayscale convert it to color
    if len(np.shape(ori_img)) == 2:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
    objects, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    for i, cnt in enumerate(objects):
        cv2.drawContours(ori_img, objects, i, (255, 102, 255), -1, lineType=8, hierarchy=hierarchy)


    return objects, hierarchy
