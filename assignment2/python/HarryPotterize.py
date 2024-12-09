#
# HarryPotterize.py - script for Q3.9, warps a Harry Potter cover onto a
#   textbook on top of a desk by calculating a planar homography
#
# Jessie Li, CS 83/183 Winter 2024
#

import cv2

# import skimage.io
# import skimage.color
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# import matplotlib.pyplot as plt

# Note:
#   skimage.transform.warp is backward (H for desk ---> book)
#   cv2.warpPerspective is forward (H for book ---> desk)

# 1. Read cv_cover.jpg, cv_desk.png, and hp_cover.jpg
cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")
hp_cover = cv2.imread("../data/hp_cover.jpg")

print("Shape of cover:", cv_cover.shape)
print("Shape of Harry Potter:", hp_cover.shape)

hp_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

print("Resized shape of Harry Potter:", hp_resized.shape)

# 2. Compute a homography using matchPics and computeH_ransac

# cv2
matches, locs1, locs2 = matchPics(cv_desk, cv_cover)

# skimage
# matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

x1 = locs1[matches[:, 0], :]
x2 = locs2[matches[:, 1], :]
H2to1, inliers = computeH_ransac(x1, x2)

# cv2, comparison
H_cv, status = cv2.findHomography(x2, x1, cv2.RANSAC)

# skimage, comparison
# H, status = cv2.findHomography(x1, x2, cv2.RANSAC)

# 3. Use the computed homography to warp hp_cover.jpg to the dimensions of cv desk.png
h, w, _ = cv_desk.shape

# cv2
warped = cv2.warpPerspective(hp_resized.swapaxes(0, 1), H2to1, (h, w)).swapaxes(0, 1)
warped_cv = cv2.warpPerspective(hp_resized.swapaxes(0, 1), H_cv, (h, w)).swapaxes(0, 1)

# skimage
# warped = skimage.transform.warp(hp_resized.swapaxes(0, 1), H2to1, output_shape=(w, h)).swapaxes(0, 1)
# warped_cv = skimage.transform.warp(hp_resized.swapaxes(0, 1), H, output_shape=(w, h)).swapaxes(0, 1)

composite = compositeH(H2to1, hp_resized, cv_desk)
composite_cv = compositeH(H_cv, hp_resized, cv_desk)

# 4. Display the images

# Plot matches and inliers
# inliers_x1 = x1[inliers]
# inliers_x2 = x2[inliers]
# inliers_matches = np.vstack((inliers.nonzero(), inliers.nonzero())).T

# fig, ax = plt.subplots()
# skimage.feature.plot_matches(ax, cv_cover, cv_desk, locs1, locs2, matches, matches_color='r', only_matches=True)
# skimage.feature.plot_matches(ax, cv_cover, cv_desk, x1, x2, inliers_matches, matches_color='g', only_matches=True)
# plt.show()

# cv2
cv2.imwrite("../results/warped.png", warped)
cv2.imwrite("../results/warped_cv.png", warped_cv)

cv2.imwrite("../results/composite.png", composite)
cv2.imwrite("../results/composite_cv.png", composite_cv)

# skimage
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(warped_cv)
# ax[1].imshow(warped)
# plt.show()

# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(composite_cv)
# ax[1].imshow(composite)
# plt.show()
