import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(img_hough, nLines):
    """
        Uses Hough transform output to detect lines.
        
        Parameters:
            img_hough: Hough transform accumulator
            nLines: number of lines to return

        Returns: 
            rhos: nLines x 1 vector with row coordinates of peaks in img_hough 
            thetas: nLines x 1 vector with column coordinates of peaks in img_hough

    """

    # Suppress non-maximal cells, considering all neighbors of a pixel.
    dilated = cv2.dilate(img_hough, np.ones((3, 3))) 
    suppressed = np.where(img_hough < dilated, 0, img_hough) 

    # Top nLines: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    rhos, thetas = np.unravel_index(np.argsort(suppressed, axis=None)[-nLines:], img_hough.shape)
    return rhos, thetas