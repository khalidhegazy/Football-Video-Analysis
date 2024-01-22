# load, split and scale the maps dataset ready for training
from os import listdir
import numpy as np
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
from PIL import Image
import cv2

path = r"soccer_seg_detection/test//"
for i, filename in enumerate(listdir(path)):
    pixels = cv2.imread(path + filename)
    # split into satellite and map
    src_img, tar_img = pixels[:, :256], pixels[:, 256:]
    x_name = r"lsd_dataset\x_train\{0:0=1d}.jpg".format(i)
    y_name = r"lsd_dataset\y_train\{0:0=1d}.jpg".format(i)

    hsv = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))  # green mask to select only the field
    frame_masked = cv2.bitwise_and(src_img, src_img, mask=mask_green)
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imwrite(x_name, cv2.cvtColor(canny, cv2.COLOR_RGB2BGR))
    cv2.imwrite(y_name, cv2.cvtColor(tar_img, cv2.COLOR_RGB2BGR))
    print("photo")