import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from SeparateTeams import SeparateTeams
from Detection import Detection
from Homography import Homography


if __name__ == '__main__' :
    path = "input.mp4"
    Detection = Detection()
    ST = SeparateTeams(path)
    Homography = Homography()

    classes = ['Ball', 'Player']

    cap = cv2.VideoCapture(path)
    out = cv2.VideoWriter('output4.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

    colors = ST.clf_colors()

    while 1:
        ret, new_img = cap.read()
        #new_img = Homography.detect_lines(img)
        boxes, confidences, class_ids = Detection.detect(ret, new_img)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .2, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        players_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                if class_ids[i] != 0:
                    players_boxes.append(boxes[i])
            results = ST.separate(new_img, players_boxes)
            print(results)
        j = 0
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if class_ids[i] == 0:
                    cv2.rectangle(new_img, (x, y), (x + w, y + h), [0, 0, 0], 2)
                    cv2.putText(new_img, "Ball", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 0, 0], 2)

                elif class_ids[i] == 1:
                    if results[j] == 0:
                        cv2.rectangle(new_img, (x, y), (x + w, y + h), colors[0], 2)
                        cv2.putText(new_img, "Team A", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
                    elif results[j] == 1:
                        cv2.rectangle(new_img, (x, y), (x + w, y + h), colors[1], 2)
                        cv2.putText(new_img, "Team B", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2)
                    elif results[j] == 2:
                        cv2.rectangle(new_img, (x, y), (x + w, y + h), colors[2], 2)
                        cv2.putText(new_img, "Team C", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[2], 2)
                    j += 1

        # img = ST.masking(img)
        cv2.imshow('frame', new_img)
        # Homography.radar(img)
        out.write(new_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
