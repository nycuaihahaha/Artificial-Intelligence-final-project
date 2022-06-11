import dataset
import adaboost
import utils
import detection
import matplotlib.pyplot as plt
import cv2
import os
# Part 1: Implement loadImages function in dataset.py and test the following code.


'''print('Loading images')
trainData = dataset.loadImages('Face Mask Dataset/Train')
print(f'The number of training samples loaded: {len(trainData)}')
testData = dataset.loadImages('Face Mask Dataset/Train')
print(f'The number of test samples loaded: {len(testData)}')

print('Show the first and last images of training dataset')
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].set_title('WithMask')
ax[0].imshow(trainData[1][0], cmap='gray')
ax[1].axis('off')
ax[1].set_title('WithoutMask')
ax[1].imshow(trainData[-1][0], cmap='gray')
'''
#plt.show()

# Part 2: Implement selectBest function in adaboost.py and test the following code.
# Part 3: Modify difference values at parameter T of the Adaboost algorithm.
# And find better results. Please test value 1~10 at least.
'''print('Start training your classifier')
clf = adaboost.Adaboost(T=10)
clf.train(trainData)

clf.save('clf_300_1_10')'''
clf = adaboost.Adaboost.load('clf_300_1_10')
if not os.path.exists('result'):
    os.makedirs('result')


#dataset.loadcsv(clf)


dataset.load_and_save('maksssksksss1.png',350,117,30,73,clf)
dataset.load_and_save('maksssksksss1.png',250,119,36,73,clf)


'''print('\nEvaluate your classifier with training dataset')
utils.evaluate(clf, trainData)

print('\nEvaluate your classifier with test dataset')
utils.evaluate(clf, testData)'''

# Part 4: Implement detect function in detection.py and test the following code.
'''print('\nUse your classifier with video.gif to get the predictions (one .txt and one .png)')
detection.detect('data/detect/detectData.txt', clf)'''
