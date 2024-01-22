import cv2
import numpy as np

class Homography:
     def __init__(self):
          # Read source image.
          self.im_dst = cv2.imread('resources/frame.jpg')
          # Four corners of the book in source image
          self.pts_dst = np.array([[314, 107], [693, 107], [903, 493], [311, 491]])

          # Read destination image.
          self.im_src = cv2.imread('resources/pitch.jpg')
          # Four corners of the book in destination image.
          self.pts_src = np.array([[487, 81], [575, 81], [575, 297], [487, 297]])

          # Calculate Homography
          self.h, self.status = cv2.findHomography(self.pts_src, self.pts_dst)

          # Warp source image to destination based on homography
          self.im_out = cv2.warpPerspective(self.im_src, self.h, (self.im_dst.shape[1], self.im_dst.shape[0]), borderValue=[255, 255, 255])
          self.mask = self.im_out[:, :, 0] < 100

          self.im_out_overlapped = self.im_dst.copy()
          self.im_out_overlapped[self.mask] = [0, 0, 255]
          # Display images
          #cv2.imshow("Source Image", self.im_src)
          #cv2.imshow("Destination Image", self.im_dst)
          #cv2.imshow("Warped Source Image", self.im_out)
          #cv2.imshow("Warped", self.im_out_overlapped)
          #cv2.waitKey(0)

     def radar(self, img):
          large_img = img
          small_img = cv2.resize(self.im_src, (196, 126))
          x_offset = 542
          y_offset = 590
          x_end = x_offset + small_img.shape[1]
          y_end = y_offset + small_img.shape[0]
          large_img[y_offset:y_end, x_offset:x_end] = small_img
          cv2.imshow("Source Image", large_img)

     def update_radar(self):
          players = []

     def detect_lines(self,frame):

          hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
          mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))  # green mask to select only the field
          frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)
          gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
          _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
          canny = cv2.Canny(gray, 50, 150, apertureSize=3)
          # Hough line detection
          lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)
          # Line segment detection
          lsd = cv2.createLineSegmentDetector(0)
          lines_lsd = lsd.detect(canny)[0]

          drawn_img = lsd.drawSegments(frame, lines_lsd)
          return drawn_img

#if __name__ == '__main__' :