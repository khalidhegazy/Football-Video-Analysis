import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Separate player into two teams and referee
class Detection:

    def __init__(self):

        self.net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", "yolov3_custom_600.weights")
        self.classes = ['Ball', 'Player']

    def detect(self, ret, img):
        if ret:
            img = cv2.resize(img, (1280, 720))
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)
            output_layers_name = self.net.getUnconnectedOutLayersNames()
            layer_outputs = self.net.forward(output_layers_name)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)
            return [boxes, confidences, class_ids]