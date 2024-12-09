#
# LucasKanade.py - implements Lucas-Kanade forward additive alignment with translation
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq


def LucasKanade(It, It1, rect):
    """
    Question 3.1
    Lucas-Kanade Forward Additive Alignment with Translation

    Computes the optimal local motion represented by only translation
    (motion in x and y directions) from frame It to frame It+1 that
    minimizes the sum of squared differences.

    Input:
        It: template image
        It1: current image
        rect: current position of the object
            (top left, bot right coordinates: x1, y1, x2, y2)

    Output:
        p: movement vector dx, dy

    """

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # put your implementation here
    # print("Shape of It:", It.shape)
    # print("Shape of It1:", It1.shape)

    h, w = It1.shape
    th, tw = It.shape

    spline_template = RectBivariateSpline(np.arange(h), np.arange(w), It)
    spline_image = RectBivariateSpline(np.arange(th), np.arange(tw), It1)

    y = np.arange(y1, y2 + 1)
    x = np.arange(x1, x2 + 1)

    # x and y values of each pixel in template
    x_temp, y_temp = np.meshgrid(x, y)
    x_temp = x_temp.reshape(-1)
    y_temp = y_temp.reshape(-1)

    template = spline_template.ev(y_temp, x_temp)
    # print("Pixels in template:", template.shape)
    print("\n")

    for _ in range(maxIters):

        # warp, or translate by p
        y_image = y_temp + p[1]
        x_image = x_temp + p[0]

        # error image
        error = template - spline_image.ev(y_image, x_image)
        error = error.reshape(-1, 1)
        # print("Shape of error", error.shape)

        # image gradients and jacobian
        Ix = spline_image.ev(y_image, x_image, dx=1, dy=0)
        Iy = spline_image.ev(y_image, x_image, dx=0, dy=1)
        # print("Shape of Ix:", Ix.shape)
        # print("Shape of Iy:", Iy.shape)

        jacobian = np.column_stack((Iy, Ix))
        # print("Shape of Jacobian:", jacobian.shape)

        # compute Hessian matrix
        hessian = jacobian.T @ jacobian
        # print("Shape of Hessian:", hessian.shape)
        # print("\n")

        # solve linear system for dp
        dp, _ = lstsq(hessian, jacobian.T @ error)[:2]
        # print("dp:", dp.reshape(-1))
        # print("\n")

        if np.linalg.norm(dp) < threshold:
            break

        # update parameters
        p += dp.reshape(-1)

    return p
