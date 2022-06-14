import argparse
from itertools import count
import ntpath
import time
import os
from unittest import result

import cv2
import imutils
from imutils import paths
import mtcnn
import numpy as np
from imutils.video import WebcamVideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# 初始化臉部偵測模型
detector = mtcnn.MTCNN()


# 偵測是否有戴口罩
def predict_mask(frame, mask_net):
    faces = []
    preds = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb, (224, 224))

    face = img_to_array(face)
    face = preprocess_input(face)
    faces.append(face)
    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return preds[0]


def main():
    # 初始化Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="mask_detector.model", help="path to the trained mask model")
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())

    maskNet = load_model(args["model"])

    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []
    count = 0
    withmask = 0
    wm = 0
    wom= 0
    without = 0
    for imagePath in imagePaths:
        count=count+1
        labe = ntpath.normpath(imagePath).split(os.path.sep)[-2]
        img = cv2.imread(imagePath)
        img = np.array(img)
        frame = imutils.resize(img, width=400)

        pred = predict_mask(frame, maskNet)
        (mask, withoutMask) = pred
        if labe == "with_mask":
            wm=wm+1
            if mask>withoutMask:
                withmask=withmask+1
        elif labe == "without_mask":
            wom=wom+1
            if ~(mask>withoutMask):
                without=without+1
    print("result")
    print(withmask)
    print(without)
    print(wm)
    print(wom)
    print((withmask+without)/(wm+wom))

if __name__ == '__main__':
    main()
