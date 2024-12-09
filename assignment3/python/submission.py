#
# submission.py - submission functions
#
# Jessie Li, CS 83/183 Winter 2024
#


import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import rq
from helper import refineF

def eight_point(pts1, pts2, M):
    """
    Question 2.1 Eight Point Algorithm
    
    Parameters:
        pts1: points in image 1 (Nx2 matrix)
        pts2: points in image 2 (Nx2 matrix)
        M: scalar value computed as max(H1, W1)
        
    Returns:
        F: fundamental matrix (3x3 matrix)

    """

    # 0. Normalize
    pts1 = pts1 / M
    pts2 = pts2 / M

    # 1. Construct N x 9 matrix A
    n = len(pts1)

    A = np.zeros((n, 9))
    # Extract x and y coordinates from pts1 and pts2
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    # Create the matrix A
    A = np.vstack([
        x1 * x2,
        x1 * y2,
        x1,
        y1 * x2,
        y1 * y2,
        y1,
        x2,
        y2, 
        np.ones_like(x1)
    ]).T

    # print(f'Shape of A: {A.shape}\n')

    # 2. Find SVD of A
    U, S, Vt = np.linalg.svd(A)

    # 3. Entries of F are the elements of column of V corresponding to the least singular value
    F = Vt[-1].reshape((3,3))

    # 4. Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # 5. Un-normalize F
    F = refineF(F, pts1, pts2)
    
    t = np.diag([1/M, 1/M, 1])
    F = t.T @ F @ t
                   
    return F

def epipolar_correspondences(im1, im2, F, pts1):
    """
    Question 2.2 Epipolar Correspondences

    Based on epipolarMatchGUI in helper.py.
    
    [I]
        im1: image 1 (H1xW1 matrix)
        im2: image 2 (H2xW2 matrix)
        F: fundamental matrix from image 1 to image 2 (3x3 matrix)
        pts1: points in image 1 (Nx2 matrix)
        
    [O]
        pts2: points in image 2 (Nx2 matrix)
        
    """

    sy, sx, _ = im2.shape
    pts2 = np.zeros_like(pts1)
    window = 17

    for j in range(len(pts1)):
        x1 = pts1[j, 0]
        y1 = pts1[j, 1]

        # Find epipolar line
        p = np.array([[x1], [y1], [1]])
        line = F @ p

        scale = np.linalg.norm(line[:2])

        if scale == 0:
            raise Exception('Zero line vector in displayEpipolar')

        line /= scale

        # Points on the epipolar line, based on epipolarMatchGUI in helper.py
        if line[0] != 0:
            xs = 0
            xe = sx - 1
            ys = -(line[0] * xs + line[2]) / line[1]
            ye = -(line[0] * xe + line[2]) / line[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(line[1] * ys + line[2]) / line[0]
            xe = -(line[1] * ye + line[2]) / line[0]

        # Max of horizontal and vertical distance between endpoints
        d = max(abs(xe - xs), abs(ye - ys))
        x_range = np.linspace(xs, xe, d)
        y_range = np.linspace(ys, ye, d)

        mininum_error = np.inf

        # Cut out an image patch in img centered at x1, y1
        im1_patch = image_patch(im1, [x1, y1], window)

        # For each point in line, compare im1_patch with each im2_patch
        for i in range(d):
            x2 = x_range[i]
            y2 = y_range[i]

            im2_patch = image_patch(im2, [int(x2), int(y2)], window)

            # Sum of squared differences as error (square of Euclidean)
            diff = im1_patch - im2_patch
            error = np.sum(diff ** 2)
            
            if error < mininum_error:
                mininum_error = error
                pts2[j, 0] = x2
                pts2[j, 1] = y2

    return pts2

def image_patch(im, point, window=17):
    """
    Helper function for epipolar_correspondences.
    Returns a patch of pixels in image `im` centered at `point`.

    [I]
        im: image
        point: center of the patch
        window: size of the patch

    [O]
        patch: image patch

    """
    h, w, _ = im.shape
    range_vals = np.int16(np.arange(-window//2, window//2 + 1))

    x = np.int16(point[0])
    y = np.int16(point[1])

    xrange = range_vals + x
    yrange = range_vals + y

    # Ensure the patch remains within image boundaries
    xrange = np.clip(xrange, 0, w - 1)
    yrange = np.clip(yrange, 0, h - 1)

    patch = im[yrange, xrange, :]

    return patch

def essential_matrix(F, K1, K2):
    """
    Question 2.3 Essential Matrix

    [I]
        F: fundamental matrix (3x3 matrix)
        K1: camera matrix 1 (3x3 matrix)
        K2: camera matrix 2 (3x3 matrix)
    
    [O]
        E: essential matrix (3x3 matrix)

    """

    return K2.T @ F @ K1

def triangulate(P1, pts1, P2, pts2):
    """
    Question 2.4 Triangulation 

    [I]
        P1: camera projection matrix 1 (3x4 matrix)
        pts1: points in image 1 (Nx2 matrix)
        P2: camera projection matrix 2 (3x4 matrix)
        pts2: points in image 2 (Nx2 matrix)
    
    [O]
        pts3d: 3D points in space (Nx3 matrix)

    """

    n = pts1.shape[0]
    pts3d = np.zeros((n, 4))

    for i in range(n):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = pts2[i, 0], pts2[i, 1]

        A = np.array([
            y1 * P1[2, :] - P1[1, :],
            P1[0, :] - x1 * P1[2, :],
            y2 * P2[2, :] - P2[1, :],
            P2[0, :] - x2 * P2[2, :]])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[-1]
        
        pts3d[i, :] = X
    
    return pts3d

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    Question 3.1 Image Rectification

    [I] 
        K1, K2: camera matrices (3x3 matrix)
        R1, R2: rotation matrices (3x3 matrix)
        t1, t2: translation vectors (3x1 matrix)
    
    [O] 
        M1, M2: rectification matrices (3x3 matrix)
        K1p, K2p: rectified camera matrices (3x3 matrix)
        R1p, R2p: rectified rotation matrices (3x3 matrix)
        t1p, t2p: rectified translation vectors (3x1 matrix)

    """
    # 1. Compute the optical centers c1 and c2 of each camera
    K1_R1_inv = np.linalg.inv(K1 @ R1)
    K2_R2_inv = np.linalg.inv(K2 @ R2)

    c1 = -K1_R1_inv @ (K1 @ t1)
    c2 = -K2_R2_inv @ (K2 @ t2)

    # 2. Compute the new rotation matrix R
    t = c1 - c2
    r1 = t / np.linalg.norm(t)
    r2 = np.array([-t[1], t[0], 0]) / np.sqrt(t[0]**2 + t[1]**2)
    r3 = np.cross(r1, r2)

    R = R1p = R2p = np.vstack((r1, r2, r3))

    # 3. Compute the new intrinsic parameters
    K1p = K2p = K2

    # 4. Compute the new translation vectors
    t1p = -R @ c1
    t2p = -R @ c2

    # 5. Finally, compute the rectification matrices of the cameras
    M1 = (K1p @ R1p) @ K1_R1_inv
    M2 = (K2p @ R2p) @ K2_R2_inv

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

def get_disparity(im1, im2, max_disp, win_size):
    """
    Question 3.2 Disparity Map

    [I] 
        im1: image 1 (H1xW1 matrix)
        im2: image 2 (H2xW2 matrix)
        max_disp: scalar maximum disparity value
        win_size: scalar window size value
    
    [O] 
        dispM: disparity map (H1xW1 matrix)

    """
    
    dispM = np.zeros((im1.shape[0], im1.shape[1], max_disp+1))

    dispM[:, :, 0] = convolve2d(np.square(im1 - im2), np.ones((win_size, win_size)), mode='same')

    for d in range(1, max_disp + 1):
        window_sum = convolve2d(
            np.square(im1[:, :-d] - im2[:, d:]), 
            np.ones((win_size, win_size)), 
            mode='same')
        
        dispM[:, d:, d] = window_sum

    dispM = np.argmin(dispM, axis=2)

    return dispM

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    Question 3.3 Depth Map

    [I] 
        dispM: disparity map (H1xW1 matrix)
        K1, K2: camera matrices (3x3 matrix)
        R1, R2: rotation matrices (3x3 matrix)
        t1, t2: translation vectors (3x1 matrix)
    
    [O] 
        depthM: depth map (H1xW1 matrix)

    """
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2

    b = np.linalg.norm(c1 - c2)
    f = K1[1, 1]

    dispM_inv = np.where(dispM == 0, 0, 1/dispM)
    return f * b * dispM_inv

def estimate_pose(x, X):
    """
    Question 4.1: Camera Matrix Estimation

    [I] 
        x: 2D points (Nx2 matrix)
        X: 3D points (Nx3 matrix)
    
    [O] 
        P: camera matrix (3x4 matrix)

    """
    # Compute the centroid of the points
    meanx = np.mean(x, axis=0)
    meanX = np.mean(X, axis=0)

    # Shift the origin of the points to the centroid
    x_translated = x - meanx
    X_translated = X - meanX

    # Normalize the points so that the largest distance from the origin is equal to 
    # sqrt(2) for 2D points or sqrt(3) for 3D points.
    d1 = np.max(np.linalg.norm(x_translated, axis=1))
    d2 = np.max(np.linalg.norm(X_translated, axis=1))

    scalex = 2 ** 0.5 / d1
    scaleX = 3 ** 0.5 / d2

    x_normalized = x_translated * scalex
    X_normalized = X_translated * scaleX

    # Similarity transform x
    Tx = np.array([[scalex, 0, -scalex * meanx[0]],
                   [0, scalex, -scalex * meanx[1]],
                   [0, 0, 1]])

    # Similarity transform X
    TX = np.array([[scaleX, 0, 0, -scaleX * meanX[0]],
                   [0, scaleX, 0, -scaleX * meanX[1]],
                   [0, 0, scaleX, -scaleX * meanX[2]],
                   [0, 0, 0, 1]])

    # Compute P
    n = x.shape[0]
    A = np.zeros((2 * n, 12))

    Xh = np.hstack((X_normalized, np.ones((X_normalized.shape[0], 1))))

    for i in range(n):
        A[2*i, :4] = Xh[i, :]
        A[2*i, 8:] = -x_normalized[i, 0] * Xh[i, :]
        A[2*i+1, 4:8] = Xh[i, :]
        A[2*i+1, 8:] = -x_normalized[i, 1] * Xh[i, :]

    # Solve for the homogeneous solution h using the SVD
    _, _, Vt = np.linalg.svd(A)

    # Extract the solution vector (the last column of V)
    P_normalized = Vt[-1].reshape(3, 4)

    # Denormalize the camera matrix
    P = np.linalg.inv(Tx) @ P_normalized @ TX

    return P

def estimate_params(P):
    """
    Question 4.2 Camera Parameter Estimation

    [I] 
        P: camera matrix (3x4 matrix)

    [O]
        K: camera intrinsics (3x3 matrix)
        R: camera extrinsics rotation (3x3 matrix)
        t: camera extrinsics translation (3x1 matrix)

    """

    # 1. Compute camera center c (heterogeneous)
    _, _, Vt = np.linalg.svd(P)
    c = Vt[-1, :3] / Vt[-1, -1]

    # 2. Compute K and R using QR decomposition
    M = P[:, :3]
    K, R = rq(M)

    # Make sure diagonal is positive: KR = (K D)(D^-1 R)
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R

    # 3. Compute the translation by t = -Rc
    t = -R @ c
    
    return K, R, t
