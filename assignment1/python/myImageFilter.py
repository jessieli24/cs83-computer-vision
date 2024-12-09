import cv2
import numpy as np

from scipy import signal, ndimage

def myImageFilter(img0, h):
    """
    Convolves an image with a given convolution filter.

    Parameters:
        img0: grayscale image
        h: convolution filter
    
    Returns:
        img1: result of convolving img0 with h

    """
    # Pad the image such that pixels lying outside the image 
    # boundary have the same intensity value as the nearest 
    # pixel that lies inside the image.
    img_height, img_width = img0.shape
    h_height, h_width = h.shape
    
    padded = np.pad(img0, ((h_height//2,), (h_width//2,)), 'edge')

    # Convolve the image with the convolution filter by applying 
    # each kernel element to an image-sized subset of pixels in the 
    # padded original.
    img1 = np.zeros(img0.shape)

    h_flipped = np.flipud(np.fliplr(h))

    for y in range(img_height):
        for x in range(img_width): 
            patch = padded[y:y+h_height, x:x+h_width]
            img1[y, x] = np.sum(np.multiply(h_flipped, patch))

    return img1
