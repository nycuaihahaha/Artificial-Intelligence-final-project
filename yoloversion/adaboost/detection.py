import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    predicts = []
    r = []
    with open(dataPath) as f:
        lines = f.readlines()
        for line in lines:
            r.append(line.replace('\n', ' ').split())

    video = cv2.VideoCapture('data/detect/video.gif')

    for x in range(50):
        _, frame = video.read()
        predict = []
        for i in range(1, len(r)):
            pic = crop(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], frame)
            p = clf.classify(cv2.resize(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY), (36, 16)))
            if i == len(r) - 1:
                predict.append(str(p))
            else:
                predict.append(str(p) + ' ')
            if p == 1:
                #cv2.rectangle(frame, (int(r[i][0]), int(r[i][1])), (int(r[i][6]), int(r[i][7])),(0, 0, 255), 3)
                cv2.line(frame, (int(r[i][0]), int(r[i][1])), (int(r[i][2]), int(r[i][3])),(0, 255, 0), 2)
                cv2.line(frame, (int(r[i][0]), int(r[i][1])), (int(r[i][4]), int(r[i][5])),(0, 255, 0), 2)
                cv2.line(frame, (int(r[i][6]), int(r[i][7])), (int(r[i][2]), int(r[i][3])),(0, 255, 0), 2)
                cv2.line(frame, (int(r[i][6]), int(r[i][7])), (int(r[i][4]), int(r[i][5])),(0, 255, 0), 2)
        predicts.append(predict)
        if x == 0:
            cv2.imwrite('data/first_frame.png', frame)

    file = 'Adaboost_pred.txt'
    f = open(file, 'w')
    f.close()

    with open(file, 'w') as f:
        for line in predicts:
            f.writelines(line)
            f.write('\n')


        # cv2.imshow('Example - Show image in window', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # End your code (Part 4)
