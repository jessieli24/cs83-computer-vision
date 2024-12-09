import math 
import cv2

import numpy as np
from scipy import signal    # For signal.gaussian function

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    """
    Finds edge intensity and orientation in an image.
    
    Parameters: 
        img0: grayscale image
        sigma: standard deviation of the Gaussian smoothing kernel
    
    Returns:
        img1: edge magnitude image

    """
    # Use convolution function to smooth out the image with the 
    # specified Gaussian kernel.

    kernel_width = 2 * math.ceil(3 * sigma) + 1
    gaussian = signal.gaussian(kernel_width, sigma).reshape(1, -1)

    # Gaussian kernel can be separated into a column filter
    # and a row filter. Applying the two 1-D filters is  
    # faster than a single application of the 2-D. 
    img_blurred = myImageFilter(myImageFilter(img0, gaussian.T), gaussian)

    # cv2.imwrite('../test/myEdgeFilter_blurred.png',  255 * img_blurred/img_blurred.max())

    # To find the image gradient in the x and y directions, convolve 
    # the smoothed image with the x- and y-oriented Sobel filters.
    sobel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), np.float32)
    sobel_y = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]), np.float32)

    imgx = myImageFilter(img_blurred, sobel_x)
    imgy = myImageFilter(img_blurred, sobel_y)

    # cv2.imwrite('../test/myEdgeFilter_imgx.png', 255 * imgx / imgx.max())
    # cv2.imwrite('../test/myEdgeFilter_imgy.png',  255 * imgy / imgy.max())

    # Calculate gradient magnitudes.
    img1 = np.sqrt(imgx**2 + imgy**2)
    # cv2.imwrite('../test/myEdgeFilter_mag.png',  255 * img1 / img1.max())

    # Calculate gradient angles.
    angles = np.rad2deg(np.arctan2(imgy, imgx))
    angles[angles < 0] += 180

    # Make edges a single pixel wide with non-maximum suppression.
    img1 = nms(img1, angles)

    return img1 / img1.max()

def nms(img, angle):
    """
    Uses maximum non-suppression to thin edges on an image.
    Vectorized to improve run-time.  

    For each pixel, look at the two neighboring pixels along the 
    gradient direction and if either of those pixels has a larger 
    gradient magnitude then set the edge magnitude at the center 
    pixel to zero. Map gradient angle to the closest of 0◦, 45◦, 
    90◦, and 135◦.

    """

    dilated_0 = cv2.dilate(img, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8))
    dilated_45 = cv2.dilate(img, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8))
    dilated_90 = cv2.dilate(img, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8))
    dilated_135 = cv2.dilate(img, np.eye(3, dtype=np.uint8))
    
    edges = np.where(((img >= dilated_0) & ((angle < 22.5) | (angle >= 157.5)))
                    | ((img >= dilated_45) & (angle >= 22.5) & (angle < 67.5))
                    | ((img >= dilated_90) & (angle >= 67.5) & (angle < 112.5)) 
                    | ((img >= dilated_135) & (angle >= 112.5) & (angle < 157.5)), img, 0)

    return edges

def nms_slow(img, gradient_angles):
    """
    Uses maximum non-suppression thin edges on an image.
    Initial implementation with two for loops.

    For each pixel, look at the two neighboring pixels along the 
    gradient direction and if either of those pixels has a larger 
    gradient magnitude then set the edge magnitude at the center 
    pixel to zero. Map gradient angle to the closest of 0◦, 45◦, 
    90◦, and 135◦.

    """
    suppressed_image = np.zeros(img.shape)
    h, w = img.shape

    for i in range(1, h-1):
        for j in range(1, w-1):

            # 0 degrees
            if gradient_angles[i, j] < 22.5 or 157.5 <= gradient_angles[i,j]:
                maximum_gradient = max(img[i, j-1], img[i, j], img[i, j+1])

            # 45 degrees
            elif 22.5 <= gradient_angles[i, j] < 67.5:
                maximum_gradient = max(img[i-1, j+1], img[i, j], img[i+1, j-1])

            # 90 degrees
            elif 67.5 <= gradient_angles[i, j] < 112.5:
                maximum_gradient = max(img[i-1, j], img[i, j], img[i+1, j])

            # 135 degrees
            else:
                maximum_gradient = max(img[i-1, j-1], img[i, j], img[i+1, j+1])

            if img[i, j] == maximum_gradient:
                suppressed_image[i, j] = img[i, j]
    
    return suppressed_image

                
def test():
    
    file = "../data/img01.jpg"
        
    # read in image
    img = cv2.imread(file)
        
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img = np.float32(img) / 255
        
    # actual Hough line code function calls
    img_edge = myEdgeFilter(img, 2)

    # everything below here just saves the outputs to files
    fname = '../test/img01_01edge.png'
    cv2.imwrite(fname, 255 * np.sqrt(img_edge / img_edge.max()))
