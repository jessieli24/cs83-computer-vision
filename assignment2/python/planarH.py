#
# planarH.py - calculates planar homographies
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
import cv2


def computeH(x1, x2):
    """
    Q3.6
    Estimates the planar homography from a set of matched point pairs.

    Parameters:
                x1, x2: Nx2 matrices with (x, y) coordinates of point pairs

    Returns:
                H2to1: 3x3 matrix for the least-squares homography from
                image 2 to image 1.

    """
    # Compute the homography between two sets of points
    n = x1.shape[0]

    A = np.zeros((2 * n, 9))

    # Calculate A (Ah = 0)
    xi_1 = x1[:, 0]
    yi_1 = x1[:, 1]

    xi_2 = x2[:, 0]
    yi_2 = x2[:, 1]

    A[::2, 0] = -xi_2
    A[::2, 1] = -yi_2
    A[::2, 2] = -1
    A[::2, 6] = xi_2 * xi_1
    A[::2, 7] = yi_2 * xi_1
    A[::2, 8] = xi_1

    A[1::2, 3] = -xi_2
    A[1::2, 4] = -yi_2
    A[1::2, 5] = -1
    A[1::2, 6] = xi_2 * yi_1
    A[1::2, 7] = yi_2 * yi_1
    A[1::2, 8] = yi_1

    U, S, Vt = np.linalg.svd(A)
    H2to1 = Vt[-1].reshape((3, 3))

    return H2to1

    # eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    # min_index = np.argmin(eigenvalues)
    # H2to1 = eigenvectors[:, min_index]
    # return H2to1.reshape((3, 3))


def computeH_norm(x1, x2):
    """
    Q3.7

    Homography on normalized data points.

    """
    # Compute the centroid of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_translated = x1 - mean1
    x2_translated = x2 - mean2

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    d1 = np.max(np.linalg.norm(x1_translated, axis=1))
    d2 = np.max(np.linalg.norm(x2_translated, axis=1))

    scale1 = 2**0.5 / d1
    scale2 = 2**0.5 / d2

    x1_normalized = x1_translated * scale1
    x2_normalized = x2_translated * scale2

    # Similarity transform 1
    T1 = np.array(
        [[scale1, 0, -scale1 * mean1[0]], [0, scale1, -scale1 * mean1[1]], [0, 0, 1]]
    )

    # Similarity transform 2
    T2 = np.array(
        [[scale2, 0, -scale2 * mean2[0]], [0, scale2, -scale2 * mean2[1]], [0, 0, 1]]
    )

    # Compute homography
    H2to1_normalized = computeH(x1_normalized, x2_normalized)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1_normalized @ T2

    return H2to1


def computeH_ransac(x1, x2):
    """
    Q3.8
    Implements RANSAC for computing a homography.

    Parameters:
        x1, x2: Nx2 matrices with the matched points

    Returns:
                bestH2to1: homography with the most inliers
                inliers: Nx1 vector with a 1 at matches in consensus set, 0 otherwise

    """

    # Compute the best fitting homography given a list of matching points

    # Parameters
    iter = 200  # number of samples
    s = 4  # number of sampled points, minimum number needed to fit the model
    d = 1  # distance threshold

    inliers = None
    bestH2to1 = None

    n = len(x1)
    if n < s:
        print("Warning: Not enough matches to calculate the homography.")
        return None, None

    x1_homogeneous = np.hstack((x1, np.ones((n, 1))))
    x2_homogeneous = np.hstack((x2, np.ones((n, 1))))

    np.random.seed(0)
    for i in range(iter):
        # 1. Sample (randomly) the number of points required to fit the model
        sample_index = np.random.choice(n, size=s, replace=False)

        x1_sample = x1[sample_index, :]
        x2_sample = x2[sample_index, :]

        # 2. Compute H using DLT
        H_sample = computeH_norm(x1_sample, x2_sample)

        # 3. Score by the fraction of inliers within preset threshold
        x1_predicted = H_sample @ x2_homogeneous.T
        x1_predicted /= x1_predicted[2, :]

        distances = np.linalg.norm(x1_homogeneous - x1_predicted.T, axis=1)
        inliers_sample = distances <= d

        # 4. Keep H if largest number of inliers
        if (inliers is None and bestH2to1 is None) or np.sum(inliers_sample) > np.sum(
            inliers
        ):
            bestH2to1 = H_sample
            inliers = inliers_sample

    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    """
        Creates a composite image after warping the template image on top
        of the image using the homography.

        Note: The homography we compute is from the image to the template.
        x_template = H2to1 * x_photo

        For warping the template to the image, we need to invert it.

    Parameters:
        H2to1: 3x3 homography matrix
        template: template image on top
        img: base image below template

    Returns:
        composite_img: composite image

    """

    # Create mask of same size as template
    mask = np.ones_like(template)

    h, w, _ = img.shape

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask.swapaxes(0, 1), H2to1, (h, w)).swapaxes(0, 1)

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(
        template.swapaxes(0, 1), H2to1, (h, w)
    ).swapaxes(0, 1)

    # Use mask to combine the warped template and the image
    composite_img = img * (1 - warped_mask) + warped_template

    return composite_img
