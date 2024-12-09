#
# q3_4.py - small test program for the `matchPics` function
#
# Jessie Li, CS 83/183 Winter 2024
#

import cv2
from matchPics import matchPics
from helper import plotMatches
import matplotlib.pyplot as plt

cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# Display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
plt.savefig("../results/matches.png", bbox_inches="tight")
