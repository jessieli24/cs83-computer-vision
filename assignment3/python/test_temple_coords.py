#
# test_temple_coords.py - generates a full 3D reconstruction of a temple based on two images,
#   some point correspondences, and intrinsics for both cameras (Q2.5)
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
import submission as sub
import helper as hlp
import matplotlib.pyplot as plt


def reprojection_error(P, pts2d, pts3d):
    """
    Calculates the reprojection error by projecting the estimated
    3D points `pts3d` back to image and computing the mean Euclidean error
    between projected 2D points and the given `pts2d`.

    [I]
        P: projection matrix
        pts2d: given points in 2D
        pts3d: calculated 3D world points

    [0]
        error: mean reprojection error for camera 1

    """

    pts2d_hat = (P @ pts3d.T).T
    pts2d_hat /= pts2d_hat[:, -1][:, np.newaxis]  # normalize
    pts2d_hat = pts2d_hat[:, :2]  # convert to heterogeneous

    errors = np.sqrt(np.sum((pts2d - pts2d_hat) ** 2, axis=1))
    return np.mean(errors)


def graph_reconstruction(pts3d):
    """
    Plot a scatterplot of 3D point correspondences from
    triangulating `pts1` and `P1` with `pts2` and `P2`.

    [I]
        pts3d: 3D points

    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = pts3d[:, 0]
    y = pts3d[:, 1]
    z = pts3d[:, 2]

    ax.scatter(x, y, z, s=2)
    ax.set_zlim(3, 8)
    # ax.set_xlim(-2, 1.5)
    # ax.set_ylim(-1, 2)
    ax.set_ylim(-0.75, 1.25)
    ax.set_ylim(-0.75, 0.75)

    ax.grid(False)

    plt.show()


# 1. Load the two temple images and the points from data/some_corresp.npz
im1 = plt.imread("../data/im1.png")
im2 = plt.imread("../data/im2.png")

data = np.load("../data/some_corresp.npz")
p1 = data["pts1"]
p2 = data["pts2"]

print("\nQuestion 2.1: Eight Point Algorithm ------------------------------------")
print("------------------------------------------------------------------------\n")

# 2. Run eight point algorithm to compute F
F = sub.eight_point(p1, p2, max(im1.shape))
print(f"\nFundamental matrix F:\n{F}\n")
# hlp.displayEpipolarF(im1, im2, F)

print("Question 2.2: Epipolar Correspondences ---------------------------------")
print("------------------------------------------------------------------------\n")
# hlp.epipolarMatchGUI(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load("../data/temple_coords.npz")
pts1 = data["pts1"]

# 4. Run epipolar_correspondences to get points in image 2
pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)

# print("Shape of pts1:", pts1.shape)
# print("Shape of pts2:", pts2.shape, "\n")

print("Question 2.3: Essential Matrix -----------------------------------------")
print("------------------------------------------------------------------------\n")
# 5. Compute the camera projection matrix P1
intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics["K1"]
K2 = intrinsics["K2"]

E = sub.essential_matrix(F, K1, K2)
print(f"Essential matrix E:\n{E}\n")

print("Question 2.4: Triangulation --------------------------------------------")
print("------------------------------------------------------------------------\n")
# Assume no rotation or translation for P1, so extrinic matrix is just [I | 0]
M1 = np.eye(3, 4)
P1 = K1 @ M1

# 6. Get 4 possible extrinsic matrices for P2
M2s = hlp.camera2(E)

# 7. Triangulate with the projection matrices
M2 = None
pts3d = None

for i in range(4):
    M2_curr = M2s[:, :, i]
    pts3d_curr = sub.triangulate(P1, pts1, K2 @ M2_curr, pts2)

    if np.all(pts3d_curr[:, 2] > 0):
        M2 = M2_curr
        pts3d = pts3d_curr

# 8. Calculate P2 with the correct extrinsic matrix
P2 = K2 @ M2

print(f"P1:\n{P1}\n")
print(f"P2:\n{P2}\n")

print(f"Share of pts3d: {pts3d.shape}\n")

# Calculate reprojection error
error1 = reprojection_error(P1, pts1, pts3d)
print(f"P1 mean Euclidean reprojection error: {error1}")

error2 = reprojection_error(P2, pts2, pts3d)
print(f"P2 mean Euclidean reprojection error: {error2}\n")

# 9. Scatter plot the correct 3D points
graph_reconstruction(pts3d)

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
np.savez(
    "../data/extrinsics.npz", R1=M1[:, 0:3], R2=M2[:, 0:3], t1=M1[:, 3], t2=M2[:, 3]
)
results = np.load("../data/extrinsics.npz")
print("Saved results:\n", results.files)
