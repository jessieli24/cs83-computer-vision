#
# panaroma.py - constructs a simple panaroma from two images
#
# Jessie Li, CS 83/183 Winter 2024
#

import cv2
from python.matchPics import matchPics
from python.planarH import computeH_ransac

image1 = cv2.imread("my_left.png")
image2 = cv2.imread("my_right.png")

matches, locs1, locs2 = matchPics(image1, image2)

x1 = locs1[matches[:, 0], :]
x2 = locs2[matches[:, 1], :]

H2to1, _ = computeH_ransac(x1, x2)

result = cv2.warpPerspective(
    image2.swapaxes(0, 1), H2to1, (image1.shape[0], image1.shape[1] + image2.shape[1])
).swapaxes(0, 1)
cv2.imwrite("../results/pano_warp.png", result)

result[0 : image1.shape[0], 0 : image1.shape[1]] = image1

cv2.imwrite("../results/panaroma.png", result)
