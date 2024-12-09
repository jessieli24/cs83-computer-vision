#
# ar.py - warps the first video ar_source.mov onto the textbook in a second video book.mov.
#
# Jessie Li, CS 83/183 Winter 2024
#

import cv2
from loadVid import loadVid
from python.matchPics import matchPics
from python.planarH import computeH_ransac, compositeH

# Write script for Q4.1x
cv_cover = cv2.imread("../data/cv_cover.jpg")

print("Loading frames...")
book_frames = loadVid("../data/book.mov")
panda_frames = loadVid("../data/ar_source.mov")

print("Shape of cover:", cv_cover.shape)
print("Shape of book frames:", book_frames.shape)
print("Shape of panda frames:", panda_frames.shape, "\n")

# Shape of book frames: (641, 480, 640, 3)
# Shape of panda frames: (511, 360, 640, 3), each frame is 360 x 640

cover_h, cover_w, _ = cv_cover.shape
panda_h, panda_w, _ = panda_frames.shape[1:]

# Real height ~270, measured in Preview (minus black borders)
new_height = 270
# Assuming panda_w > panda_h
new_width = new_height * cover_w / cover_h

# Crop the panda frames to match the aspect ratio of cv_cover
panda_frames = panda_frames[
    :,
    int((panda_h - new_height) // 2) : int((panda_h + new_height) // 2),
    int((panda_w - new_width) // 2) : int((panda_w + new_width) // 2),
    :,
]
print("Shape of cropped panda frames:", panda_frames.shape, "\n")

# For saving the video
# https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "ar.avi", fourcc, 20.0, (book_frames.shape[2], book_frames.shape[1])
)

# MP4 output (for Mac QuickTime Player, can't open AVI)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('../results/ar.mp4', fourcc, 20.0, (book_frames.shape[2], book_frames.shape[1]))

# for i in range(10):
for i in range(min(len(book_frames), len(panda_frames))):

    book_frame = book_frames[i]
    panda_frame = panda_frames[i]

    # Resize panda frames to match size of cv_cover
    panda_frame = cv2.resize(panda_frame, (cover_w, cover_h))
    # print("Final shape of panda frame:", panda_frame.shape)

    # ORB for feature detection and description
    # https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    # https://amroamroamro.github.io/mexopencv/matlab/cv.ORB.detectAndCompute.html
    # orb = cv2.ORB_create()
    # locs1, desc1 = orb.detectAndCompute(book_frame, None)
    # locs2, desc2 = orb.detectAndCompute(cv_cover, None)
    # matches = briefMatch(desc1, desc2)

    # points1 = np.array([kp.pt for kp in locs1])
    # points2 = np.array([kp.pt for kp in locs2])

    # x1 = points1[matches[:, 0], :] # book
    # x2 = points2[matches[:, 1], :] # cover

    matches, locs1, locs2 = matchPics(book_frame, cv_cover)
    x1 = locs1[matches[:, 0], :]  # book
    x2 = locs2[matches[:, 1], :]  # cover

    H2to1, inliers = computeH_ransac(x1, x2)

    if H2to1 is None or inliers is None:
        continue

    # h, w, _ = book_frame.shape
    # warped = cv2.warpPerspective(panda_frame.swapaxes(0, 1), H2to1, (h, w)).swapaxes(0, 1)
    # cv2.imwrite(f'../video/warped{i}.png', warped)

    composite = compositeH(H2to1, panda_frame, book_frame)
    out.write(composite)
    # cv2.imwrite(f'../video/{i}.png', composite)

print("Releasing VideoWriter...")
out.release()
