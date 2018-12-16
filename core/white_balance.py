# White Balance correction function by Monica Tessman and Malia Gehan

import cv2
import os
import numpy as np

# import plantcv.params


def _hist(img, hmax, type):
    hist, bins = np.histogram(img, bins='auto')
    max1 = np.amax(bins)
    alpha = hmax / float(max1)
    corrected = np.asarray(np.where(img <= max1, np.multiply(alpha, img), hmax), type)

    return corrected



def white_balance(img):
    """
        Corrects the exposure of an image based on its histogram.
    """

    ori_img = np.copy(img)

    iy, ix, iz = np.shape(img)
    hmax = 255
    type = np.uint8
    
    mask = np.zeros((iy, ix, 3), dtype=np.uint8)

    x = 0
    y = 0
    w = ix
    h = iy
    

    cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    c1 = img[:, :, 0]
    c2 = img[:, :, 1]
    c3 = img[:, :, 2]

    channel1 = _hist(c1, hmax, type)
    channel2 = _hist(c2, hmax, type)
    channel3 = _hist(c3, hmax, type)

    finalcorrected = np.dstack((channel1, channel2, channel3))

    return finalcorrected
