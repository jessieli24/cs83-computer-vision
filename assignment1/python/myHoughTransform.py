import numpy as np

import matplotlib.pyplot as plt

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    """
    Applies the Hough Transform to an edge magnitude image.

    Parameters:
        img_threshold: edge magnitude image, thresholded to ignore pixels with a low edge filter response
        rhoRes: distance resolution of the Hough transform accumulator in pixels
        thetaRes: angular resolution of the accumulator in radians

    Returns:
        img_hough: Hough transform accumulator, contains number of “votes” for all lines 
        rhoScale: array of rho values 
        thetaScale: array of theta values
    """
    h, w = img_threshold.shape

    thetaScale = np.arange(0, 2*np.pi, thetaRes)
    sin_theta = np.sin(thetaScale)
    cos_theta = np.cos(thetaScale)

    rhoMax = np.ceil(np.sqrt(w**2 + h**2))
    rhoScale = np.arange(0, rhoMax, rhoRes)

    img_hough = np.zeros((rhoScale.size, thetaScale.size))

    y, x = np.nonzero(img_threshold)
    t = np.arange(0, thetaScale.size)

    for i in range(thetaScale.size):
        rho = x * cos_theta[i] + y * sin_theta[i]
        r = np.int16(rho/rhoRes)

        # Ignore negative r values
        r_unique, votes = np.unique(r[r >= 0], return_counts=True)

        img_hough[r_unique, i] = votes

    return img_hough, rhoScale, thetaScale
