#
# LucasKanadeAffine.py - implements Lucas-Kanade forward additive alignment with affine transformation
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq


def LucasKanadeAffine(It, It1, rect):
    """
    Question 3.2
    Lucas-Kanade Forward Additive Alignment with Affine Transformation

    Computes the optimal local motion M represented by an affine
    transformation with 6 parameters.

    Input:
        It: template image
        It1: current image
        rect: current position of the object
            (top left, bot right coordinates: x1, y1, x2, y2)

    Output:
        M: affine warp matrix [2x3 numpy array]

    """

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6, 1))
    x1, y1, x2, y2 = rect

    # put your implementation here
    ih, iw = It1.shape
    th, tw = It.shape

    # force rectangle to be in template bounds
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, tw - 1)
    y2 = min(y2, th - 1)

    # interpolating functions
    spline_template = RectBivariateSpline(np.arange(th), np.arange(tw), It)
    spline_image = RectBivariateSpline(np.arange(ih), np.arange(iw), It1)

    # x and y values of each pixel in template
    y_temp = np.arange(y1, y2 + 1)
    x_temp = np.arange(x1, x2 + 1)
    x_temp, y_temp = np.meshgrid(x_temp, y_temp)
    x_temp = x_temp.reshape(-1)
    y_temp = y_temp.reshape(-1)

    template = spline_template.ev(y_temp, x_temp)
    # print("Pixels in template:", template.shape)
    # print("\n")

    # homogeneous template points
    points_template = np.vstack((x_temp, y_temp, np.ones_like(x_temp)))

    M = np.array([[1.0], [0], [0], [0], [1.0], [0]])

    for _ in range(maxIters):
        # warp
        points_warped = (M + p).reshape(2, 3) @ points_template

        y_warp, x_warp = points_warped[1, :], points_warped[0, :]
        image_warped = spline_image.ev(y_warp, x_warp)

        # error image
        error = template - image_warped
        error = error.reshape(-1, 1)
        # print("Shape of error", error.shape)

        # image gradients
        Ix = spline_image.ev(y_warp, x_warp, dx=0, dy=1)
        Iy = spline_image.ev(y_warp, x_warp, dx=1, dy=0)
        # print("Shape of Ix:", Ix.shape)
        # print("Shape of Iy:", Iy.shape)

        # calculate Jacobian
        gradient = np.column_stack((Ix, Iy)).reshape(-1, 1)
        # print("Shape of gradient:", gradient.shape)

        zeros = np.zeros_like(points_template.T)
        dWdp = np.column_stack(
            (points_template.T, zeros, zeros, points_template.T)
        ).reshape(-1, 6)
        # print("Shape of dWdp:", dWdp.shape)

        jacobian = gradient * dWdp

        # J = Ix * [x y 1 0 0 0] + Iy * [0 0 0 x y 1]
        jacobian = jacobian[0::2, :] + jacobian[1::2, :]
        # print("Shape of Jacobian:", jacobian.shape)

        # calculate Hessian
        hessian = jacobian.T @ jacobian
        # print("Shape of Hessian:", hessian.shape)
        # print("\n")

        dp, _ = lstsq(hessian, jacobian.T @ error)[:2]
        # print("dp:", dp.reshape(-1))
        # print("\n")

        if np.linalg.norm(dp) < threshold:
            break

        # update parameters
        p += dp

    # reshape the output affine matrix
    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]]).reshape(2, 3)

    return M
