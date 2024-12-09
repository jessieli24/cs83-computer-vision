#
# LucasKanade_Pyramid.py - implements Lucas-Kanade tracking on a pyramid
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq

# https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html
def pyramids(image, levels=3):

    image_layer = image.copy() 
    pyramids = [image_layer]

    for _ in range(levels):
        image_layer = cv2.pyrDown(image_layer)
        pyramids.append(image_layer)

    return pyramids

def LucasKanade_Pyramid(It, It1, rect):
    '''
    Question 4.2
    Lucas-Kanade Tracking on an Image Pyramid

    Computes the optimal local motion represented by only translation 
    (motion in x and y directions) from frame It to frame It+1. Runs on
    an image pyramid.

    Input:
        It: template image
        It1: current image
        rect: current position of the object (top left, bot right coordinates: x1, y1, x2, y2)
        num_levels: number of levels in the image pyramid

    Output:
        p: movement vector dx, dy

    '''

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # image_height, image_width = It1.shape
    # temp_height, temp_width = It.shape
    levels = 4
    template_layers = pyramids(It)
    image_layers = pyramids(It1)

    print('Number of layers:', len(template_layers))

    for level in range(levels-1, -1, -1):
        template_pyramid = template_layers[level]
        image_pyramid = image_layers[level]
        print(f'Size of template: {template_pyramid.shape}')

        h, w = image_pyramid.shape
        th, tw = template_pyramid.shape

        spline_image = RectBivariateSpline(np.arange(th), np.arange(tw), image_pyramid)
        spline_template = RectBivariateSpline(np.arange(h), np.arange(w), template_pyramid)

        # 3 2 1 0
        s = 2 ** level
        x1p, y1p = x1 // s, y1 // s
        x2p, y2p = x2 // s, y2 // s

       # print(x1p, y1p, x2p, y2p)
        y = np.arange(y1p, y2p + 1)
        x = np.arange(x1p, x2p + 1)

        # x and y values of each pixel in template
        x_temp, y_temp = np.meshgrid(x, y)
        x_temp = x_temp.reshape(-1)
        y_temp = y_temp.reshape(-1)

        template_values = spline_template.ev(y_temp, x_temp)

        for i in range(maxIters):
            # warp, or translate by p
            y_image = y_temp + p[1]
            x_image = x_temp + p[0]

            # error image
            error = template_values - spline_image.ev(y_image, x_image)
            error = error.reshape(-1, 1)
            # print('Nonzero error count:', np.count_nonzero(error))

            # image gradients and jacobian
            Ix = spline_image.ev(y_image, x_image, dx=1, dy=0)
            Iy = spline_image.ev(y_image, x_image, dx=0, dy=1)

            jacobian = np.column_stack((Iy, Ix))

            # compute Hessian matrix
            hessian = jacobian.T @ jacobian

            # solve linear system for dp
            dp, _ = lstsq(hessian, jacobian.T @ error)[:2]

            if np.linalg.norm(dp) < threshold:
                print(f'Number of iterations: {i}')
                break

            # update parameters
            p += dp.reshape(-1)

        # scale the parameters for next layer
        if level != 0:
            p *= 2
            print(f'p for next pyramid: {p}\n')

    print(f'Final p: {p}\n')
    return p
