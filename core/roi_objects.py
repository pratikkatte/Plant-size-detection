# Find Objects Partially Inside Region of Interest or Cut Objects to Region of Interest

import cv2
import numpy as np
import os

def roi_objects(img, roi_type, roi_contour, roi_hierarchy, object_contour, obj_hierarchy):
    """Find objects partially inside a region of interest or cut objects to the ROI.

    Inputs:
    img            = RGB or grayscale image data for plotting
    roi_type       = 'cutto' or 'partial' (for partially inside)
    roi_contour    = contour of roi, output from "View and Adjust ROI" function
    roi_hierarchy  = contour of roi, output from "View and Adjust ROI" function
    object_contour = contours of objects, output from "Identifying Objects" function
    obj_hierarchy  = hierarchy of objects, output from "Identifying Objects" function

    Returns:
    kept_cnt       = kept contours
    hierarchy      = contour hierarchy list
    mask           = mask image
    obj_area       = total object pixel area

    :param img: numpy.ndarray
    :param roi_type: str
    :param roi_contour: list
    :param roi_hierarchy: numpy.ndarray
    :param object_contour: list
    :param obj_hierarchy: numpy.ndarray
    :return kept_cnt: list
    :return hierarchy: numpy.ndarray
    :return mask: numpy.ndarray
    :return obj_area: int
    """

    if len(np.shape(img)) == 3:
        ix, iy, iz = np.shape(img)
    else:
        ix, iy = np.shape(img)

    size = ix, iy, 3
    background = np.zeros(size, dtype=np.uint8)
    ori_img = np.copy(img)
    # If the reference image is grayscale convert it to color
    if len(np.shape(ori_img)) == 2:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
    w_back = background + 255
    background1 = np.zeros(size, dtype=np.uint8)
    background2 = np.zeros(size, dtype=np.uint8)


    for c, cnt in enumerate(object_contour):
        length = (len(cnt) - 1)
        stack = np.vstack(cnt)

        keep = False
        for i in range(0, length):
            pptest = cv2.pointPolygonTest(roi_contour[0], (stack[i][0], stack[i][1]), False)
            if int(pptest) != -1:
                keep = True
        if keep:
            if obj_hierarchy[0][c][3] > -1:
                cv2.drawContours(w_back, object_contour, c, (255, 255, 255), -1, lineType=8,
                                 hierarchy=obj_hierarchy)
            else:
                cv2.drawContours(w_back, object_contour, c, (0, 0, 0), -1, lineType=8, hierarchy=obj_hierarchy)
        else:
            cv2.drawContours(w_back, object_contour, c, (255, 255, 255), -1, lineType=8, hierarchy=obj_hierarchy)

    kept = cv2.cvtColor(w_back, cv2.COLOR_RGB2GRAY)
    kept_obj = cv2.bitwise_not(kept)
    mask = np.copy(kept_obj)
    obj_area = cv2.countNonZero(kept_obj)
    kept_cnt, hierarchy = cv2.findContours(kept_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    cv2.drawContours(ori_img, kept_cnt, -1, (0, 255, 0), -1, lineType=8, hierarchy=hierarchy)
    cv2.drawContours(ori_img, roi_contour, -1, (255, 0, 0), 5, lineType=8, hierarchy=roi_hierarchy)

    return kept_cnt, hierarchy, mask, obj_area
