#
# matchPics.py
#
# Jessie Li, CS 83/183 Winter 2024
#

import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2):
    """
    Detects, describes, and matches features in two images I1 and I2.

    Parameters:
        I1, I2: images

    Returns:
                matches: px2 matrix
                                                first column is indices into locs1
                                                second column is indices into locs2
                locs1: Nx2 matrix with x coordinates of feature points
                locs2: Nx2 matrix with y coordinates of feature points

    """
    # I1, I2: Images to match
    sigma = 0.15
    ratio = 0.7

    # Convert images to grayscale
    if I1.ndim == 3:
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    if I2.ndim == 3:
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Detect features in both images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
