#
# InverseCompositionAffine.py - implements inverse compositional alignment with affine transformation
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq


def InverseCompositionAffine(It, It1, rect):
    """
    Question 3.3
    Inverse Compositional Alignment with Affine Transformation

    Computes the optimal local motion M represented by an affine
    transformation with 6 parameters using inverse compositional
    alignment.

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

    template = spline_template.ev(y_temp, x_temp).reshape(-1, 1)

    # homogeneous template points
    points_template = np.vstack((x_temp, y_temp, np.ones_like(x_temp)))

    # precompute gradient of T
    Tx = spline_template.ev(y_temp, x_temp, dx=0, dy=1)
    Ty = spline_template.ev(y_temp, x_temp, dx=1, dy=0)

    # precompute Jacobian
    gradT = np.column_stack((Tx, Ty)).reshape(-1, 1)

    zeros = np.zeros_like(points_template.T)
    dWdp = np.column_stack(
        (points_template.T, zeros, zeros, points_template.T)
    ).reshape(-1, 6)

    jacobian = gradT * dWdp
    jacobian = jacobian[0::2, :] + jacobian[1::2, :]
    # print("Shape of Jacobian:", jacobian.shape)

    # precompute Hessian
    hessian = jacobian.T @ jacobian

    eye = np.eye(3)
    matrix_p = eye + np.vstack((p.reshape(2, 3), np.zeros((1, 3))))
    # print("Shape of warp matrix p (expect 3 x 3):", matrix_p.shape)

    for _ in range(maxIters):
        # print("Warp matrix p:\n", matrix_p)

        # warp
        points_warped = matrix_p @ points_template
        points_warped = points_warped / points_warped[2, :]

        y_warp, x_warp = points_warped[1, :], points_warped[0, :]
        image_warped = spline_image.ev(y_warp, x_warp).reshape(-1, 1)

        # error image
        error = image_warped - template
        # print("Nonzero error count:", np.count_nonzero(error))

        dp, _ = lstsq(hessian, jacobian.T @ error)[:2]
        # print("dp:", dp.reshape(-1))
        # print("\n")

        if np.linalg.norm(dp) < threshold:
            break

        # update parameters
        matrix_dp = eye + np.vstack((dp.reshape(2, 3), np.zeros((1, 3))))
        matrix_p = matrix_p @ np.linalg.inv(matrix_dp)

    p = (matrix_p - eye)[:2, :].reshape(-1, 1)

    # reshape the output affine matrix
    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]]).reshape(2, 3)

    return M
