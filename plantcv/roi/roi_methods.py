# ROI functions

import os
import cv2
import numpy as np



# Create a rectangular ROI
def rectangle(x, y, h, w, img):
    """Create a rectangular ROI.

    Inputs:
    x             = The x-coordinate of the upper left corner of the rectangle.
    y             = The y-coordinate of the upper left corner of the rectangle.
    h             = The height of the rectangle.
    w             = The width of the rectangle.
    img           = An RGB or grayscale image to plot the ROI on in debug mode.

    Outputs:
    roi_contour   = An ROI set of points (contour).
    roi_hierarchy = The hierarchy of ROI contour(s).

    :param x: int
    :param y: int
    :param h: int
    :param w: int
    :param img: numpy.ndarray
    :return roi_contour: list
    :return roi_hierarchy: numpy.ndarray
    """

    # Get the height and width of the reference image
    height, width = np.shape(img)[:2]

    # Create the rectangle contour vertices
    pt1 = [x, y]
    pt2 = [x, y + h - 1]
    pt3 = [x + w - 1, y + h - 1]
    pt4 = [x + w - 1, y]

    # Create the ROI contour
    roi_contour = [np.array([[pt1], [pt2], [pt3], [pt4]], dtype=np.int32)]
    roi_hierarchy = np.array([[[-1, -1, -1, -1]]], dtype=np.int32)



    return roi_contour, roi_hierarchy


