#
# project_cad.py - projects a CAD model to the image (Q4.3)
#
# Jessie Li, CS 83/183 Winter 2024
#


import numpy as np
import submission as sub
import matplotlib.pyplot as plt

# 1. Load data
data = np.load("../data/pnp.npz", allow_pickle=True)
image = data["image"]
X = data["X"]
x = data["x"]
cad = data["cad"]

# 2. Estimate P, K, R, and t
P = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(P)

# 3. Use camera matrix P to project 3D points X onto the image
Xh = np.hstack((X, np.ones((X.shape[0], 1))))
ph = P @ Xh.T

p = ph[:2, :] / ph[2, :]
p = p.T

# 4. Plot the given 2D points x and the projected 3D points
fig = plt.figure()
ax = fig.add_subplot(111)

ax.imshow(image)
ax.scatter(
    x[:, 0],
    x[:, 1],
    label="Given points",
    edgecolors="green",
    facecolors="none",
    marker="o",
    s=60,
)
ax.scatter(p[:, 0], p[:, 1], label="Projected points", c="black", marker=".")

plt.legend()
plt.title("Given Points and Projected Points")
plt.show()

# 5. Draw the CAD model rotated by estimated rotation R
cad_pts = cad["vertices"][0, 0]

rotated = R @ cad_pts.T
fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")
ax.plot(rotated[0], rotated[1], rotated[2], color="blue", linewidth=0.4)

plt.title("Rotated CAD")
plt.show()

# 6. Project all the CAD’s vertices onto the image and draw the projected
# CAD model overlapping with the 2D image.
fig = plt.figure()
ax = fig.add_subplot(111)

p_hom = np.hstack((cad_pts, np.ones((cad_pts.shape[0], 1))))
pp_hom = P @ p_hom.T

pp = pp_hom[:2, :] / pp_hom[2, :]
pp = pp.T

ax.imshow(image)
ax.plot(pp[:, 0], pp[:, 1], color="red", linewidth=4, alpha=0.4)

plt.title("Projected CAD Model and Image")
plt.show()
