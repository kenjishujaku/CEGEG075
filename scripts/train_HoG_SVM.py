### Creating HoG-SVM detector ###
import os
import cv2
import numpy as np
import pylab as plt
# import pickle
# import xml.etree.ElementTree as ET
# import re
import imutils
# import sys
import random

# move to the directory containing images for training
os.chdir('/home/kenji03/Desktop/CEGEG075/Coursework/Classifier')
# define parameters for HoG descriptor
win_size = (100,40)
block_size = (8,8)
block_stride = (4,4)
cell_size = (4,4)
bins = 9
numPos = 550
numNeg = 500
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# create a matrix filled with zeros which has a shape of
traindata = np.zeros((550+500, 7776),dtype=np.float32)
# loading all positive and negetive samples as a row into the matrix
train_hog = []
train_labels = []
# make lists of training data
pos_neg_data = os.listdir("pos")+os.listdir("neg")
# shuffle the positive and negative samples
pos_neg_data = random.sample(pos_neg_data,len(pos_neg_data))
# divide the dataset into training and testing data for cross validation
train_data = pos_neg_data[0:int((len(pos_neg_data))*0.8)]
test_data = pos_neg_data[0:int((len(pos_neg_data))*0.2)]

# train svm classifier with HoG features of positive and negative samples
for i in random.sample(train_data,len(train_data)):
    if ('pos' in i) == True:
        img = cv2.imread('pos/' + i)
        img = cv2.resize(img,win_size)
        hog_feature = hog.compute(img)
        train_hog.append(hog_feature)
        train_labels.append(1) # add 1 as positive label

    if ('neg' in i) == True:
        img = cv2.imread('neg/' + i)
        img = cv2.resize(img,win_size)
        hog_feature = hog.compute(img)
        train_hog.append(hog_feature)
        train_labels.append(0) # add 0 as negative label

train_hog = np.array(train_hog)
train_labels = np.array(train_labels,dtype=int)

# set parameters of SVM
svm = cv2.ml.SVM_create()
# set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# set SVM kernel to radial basis function
svm.setKernel(cv2.ml.SVM_LINEAR)
# set parameter C
svm.setC(0.5)
# set parameter Gamma
svm.setGamma(1)
# train SMV on training data
svm.train(train_hog, cv2.ml.ROW_SAMPLE,train_labels)
# save the trained model
svm.save('/home/kenji03/Desktop/CEGEG075/Coursework/SVM_Classifier/HoG/test.xml')
# estimate labels of training data set
train_res = svm.predict(train_hog)[1].ravel()
# calculate the correct response rate
correct = 0
for i in range(0,len(train_data)):
    if train_res[i] == train_labels[i]:
        correct += 1

print ((correct/len(train_data))*100) # correct response rate = 100%

# repeat same process as the previous process using test samples of training images
test_hog = []
test_labels = []
for i in random.sample(test_data,len(test_data)):
    if ('pos' in i) == True:
        img = cv2.imread('pos/' + i)
        img = cv2.resize(img,win_size)
        hog_feature = hog.compute(img)
        test_hog.append(hog_feature)
        test_labels.append(1)

    if ('neg' in i) == True:
        img = cv2.imread('neg/' + i)
        img = cv2.resize(img,win_size)
        hog_feature = hog.compute(img)
        test_hog.append(hog_feature)
        test_labels.append(0)

test_hog = np.array(test_hog)
test_labels = np.array(test_labels,dtype=int)
test_res = svm.predict(test_hog)[1].ravel()

# calculate the correct response rate
correct = 0
for i in range(0,len(test_data)):
    if test_res[i] == test_labels[i]:
        correct += 1
print ((correct/len(test_data))*100) # correct response rate = 100%

