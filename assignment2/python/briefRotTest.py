#
# briefRotTest.py - script for Q3.5, demonstrates how the BRIEF descriptor works with rotations
#
# Jessie Li, CS 83/183 Winter 2024
#

import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches

import scipy
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# Read the image and convert to grayscale, if necessary
image = cv2.imread("../data/cv_cover.jpg")

counts = []

for i in range(36):
    # Rotate image
    rotated_image = scipy.ndimage.rotate(image, 10 * i, reshape=False)

    # Compute features, descriptors and match features
    matches, locs1, locs2 = matchPics(rotated_image, image)

    # Update histogram with the count of matches for this orientation
    counts.append(len(matches))

    if i % 6 == 0:
        fig = plotMatches(rotated_image, image, matches, locs1, locs2)
        fig.savefig(f"../results/rotation{i}.png", bbox_inches="tight")

        print(f"Number of matches for rotation by {i * 10} degrees:", len(matches))

# Display histogram
fig = plt.figure(figsize=(12, 8))

plt.bar(np.arange(36), np.array(counts))

plt.xticks(np.arange(0, 36, 3), labels=np.arange(0, 360, 30), fontsize=10)

plt.xlabel("Rotation (degrees)", fontsize=14)
plt.ylabel("Number of matches", fontsize=14)

fig.savefig("../results/histogram.png", bbox_inches="tight")
