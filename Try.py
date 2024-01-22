# Import libraries
import cv2
import os
import numpy as np
from Detection import Detection

if __name__ == '__main__' :
    pic = cv2.imread("resources/001_AB.jpg")
    print(np.shape(pic))
    frame, map = pic[:, :256], pic[:, 256:]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))  # green mask to select only the field
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Hough line detection
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 50, None, 50, 20)
    # Line segment detection
    lsd = cv2.createLineSegmentDetector(0)
    lines_lsd = lsd.detect(canny)[0]

    drawn_img = lsd.drawSegments(frame, lines_lsd)

    gary_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gary_frame,4,0.9,10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        corner_frame = cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    cv2.imshow('frame', drawn_img)
    cv2.imshow('lines', canny)
    cv2.imshow('corners', map)
    cv2.waitKey(0)