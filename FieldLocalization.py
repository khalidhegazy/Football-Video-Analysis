import cv2
import numpy as np

class FieldLocalization:

    #def __init__(self):


    def isolate_field(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # find green pitch
        light_green = np.array([36, 25, 25])
        dark_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, light_green, dark_green)

        # removing small noises
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # apply mask over original frame
        return cv2.bitwise_and(img, img, mask=opening)

    def intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    def segmented_intersections(self, lines):
        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i + 1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(self.intersection(line1, line2))

        return intersections

    def absloute_theta(self, theta):
        if theta > 2.26:
            return np.pi - theta
        else:
            return theta

    def filter_lines(self, lines, num_lines_to_find):
        filtered_lines = np.zeros([num_lines_to_find, 1, 2])

        # Save the first line
        filtered_lines[0, 0, :] = lines[0, 0, :]
        print("Line 1: rho = %.1f theta = %.3f" % (filtered_lines[0, 0, 0], filtered_lines[0, 0, 1]))
        idx = 1  # Index to store the next unique line
        # Initialize all rows the same
        for i in range(1, num_lines_to_find):
            filtered_lines[i, 0, :] = filtered_lines[0, 0, :]

        # Filter the lines
        num_lines = lines.shape[0]
        for i in range(0, num_lines):
            line = lines[i, 0, :]
            rho = abs(line[0])
            theta = self.absloute_theta(line[1])

            # For this line, check which of the existing 4 it is similar to.
            closeness_rho = np.isclose(rho, filtered_lines[:, 0, 0], rtol=0.0, atol=50.0)  # 10 pixels
            closeness_theta = np.isclose(theta, filtered_lines[:, 0, 1], rtol=0.0, atol=np.pi / 36.0)  # 10 degrees

            similar_rho = np.any(closeness_rho)
            similar_theta = np.any(closeness_theta)
            similar = (similar_rho and similar_theta)

            if not similar:
                print("Found a unique line: %d rho = %.1f theta = %.3f" % (i, rho, theta))
                filtered_lines[idx, 0, :] = lines[i, 0, :]
                idx += 1

            if idx >= num_lines_to_find:
                print("Found %d unique lines!" % (num_lines_to_find))
                break

            return filtered_lines