import os
import cv2
import csv
import numpy as np
import adaboost

# load images from .csv and show result at the same time for each image
def loadcsv(clf):
    path = 'output1.csv'
    with open(path, encoding='utf-8', newline='') as csvfile:
        flag = 0
        mask_rec = []
        rows = csv.reader(csvfile)
        for row in rows:
            if '.png' in row[0]:

                if flag:

                    for i in mask_rec:
                        print(i)
                        cv2.rectangle(imgo, (i[0],i[1]),(i[2],i[3]), (255, 0, 0), 2)
                    cv2.imshow("mask", imgo)
                    cv2.waitKey(0)
                    flag = 1
                    img_name = row[0]
                    mask_rec = []
                else:
                    flag = 1
                    img_name = row[0]
            elif flag:
                idx = [img_name]+row
                w = int(idx[3])
                h = int(idx[4])
                imgo = cv2.imread('mask/images/' + idx[0])

                img = cv2.resize(cv2.imread('mask/images/' + idx[0], cv2.IMREAD_GRAYSCALE), (416,416))
                # int(idx[2]) - h: int(idx[2])
                # int(idx[1]) : int(idx[1]) + w
                print(idx)

                img = img[max(int(idx[2]) - int(h/2),0):int(idx[2])+int(h/2), max(int(idx[1])-int(w/2),0):int(idx[1])+int(w/2)]
                if clf.classify(cv2.resize(img, (36, 16))):
                    imgo = cv2.resize(imgo,(416,416))
                    mask_rec.append([max(int(idx[1])-int(w/2),0), max(int(idx[2]) - int(h/2),0), int(idx[1])+int(w/2), int(idx[2])+int(h/2)])

# load and save the result after classifying
def load_and_save(image_name,x,y,w,h,clf):
    for i in os.walk('result/'):
        if image_name in i[2]:
            path = 'result/'
            break
        else:
            path = 'mask/images/'

    imgo = cv2.imread(path+image_name)
    # print(imgo.shape)
    wo = imgo.shape[1]
    ho = imgo.shape[0]
    img = cv2.resize(cv2.imread(path+image_name, cv2.IMREAD_GRAYSCALE), (416, 416))

    img = img[max(int(y) - int(h / 2), 0):int(y) + int(h / 2),
          max(int(x) - int(w / 2), 0):int(x) + int(w / 2)]
    if clf.classify(cv2.resize(img, (36, 16))):
        imgo = cv2.resize(imgo, (416, 416))
        cv2.rectangle(imgo, (max(int(x) - int(w / 2), 0), max(int(y) - int(h / 2), 0)), (int(x) + int(w / 2), int(y)+int(h/ 2)), (255, 0, 0), 2)
        #cv2.imshow("mask", imgo)
        #cv2.waitKey(0)
        cv2.imwrite('result/'+image_name, cv2.resize(imgo,(wo,ho)) )














def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1
    list1 = []
    '''if dataPath == 'Face Mask Dataset/Train':
        counter = 0
        for image in os.listdir(dataPath + '/WithMask/'):
            if counter == 1000:
                break
            list1.append((cv2.resize(cv2.imread(dataPath + '/WithMask/' + image, cv2.IMREAD_GRAYSCALE), (36, 16)), 1))
            counter += 1
        counter = 0
        for image in os.listdir(dataPath + '/WithoutMask/'):
            if counter == 1000:
                break
            list1.append((cv2.resize(cv2.imread(dataPath + '/WithoutMask/' + image, cv2.IMREAD_GRAYSCALE), (36, 16)), 0))
            counter += 1
    else:'''
    for image in os.listdir(dataPath + '/WithMask/'):
        list1.append((cv2.resize(cv2.imread(dataPath + '/WithMask/' + image, cv2.IMREAD_GRAYSCALE), (36, 16)), 1))

    for image in os.listdir(dataPath + '/WithoutMask/'):
            list1.append(
                (cv2.resize(cv2.imread(dataPath + '/WithoutMask/' + image, cv2.IMREAD_GRAYSCALE), (36, 16)), 0))

    dataset = list1
    #print(list1[0])
    return dataset
    # End your code (Part 1)
    

