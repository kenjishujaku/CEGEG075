# import method.py that I created
import method as cw
# import libraries needed
import os
import csv
import cv2
import itertools
import numpy as np

# move to the directory containing train and test images
os.chdir('/home/kenji03/Desktop/CEGEG075/Coursework/CarData')
# open output file for recording the result
# f = open("/home/kenji03/Desktop/Image_Processing/Coursework/Output/adaboost_lbp/adaboost_lbp","a")
# csvWriter = csv.writer(f)

# create variables for counting correct positive and false positive, and a list of IoU accuracy
false_pos = 0
correct_pos = 0
accuracy_avg = []

# open true location text file including ground-true coordinate of test images
with open("trueLocations.txt", "r") as truelocation:
    # read the text file
    reader = csv.reader(truelocation, delimiter="t")
    # create a list for ground-truth Point
    gt_point_list = []
    # loop through the data
    for i in truelocation:
        # remove the symbol ":, (), "", - " and space
        i = i.replace(":", "")
        i = i.replace("(", "")
        i = i.replace(")", "")
        # i = i.replace(",", "")
        i = i.replace("-", "")
        i = i.strip(" ").split()
        # print (i) # e.g. ['165', '62,31', '61,137']
        # create a temporary list for the point data
        point_list = []
        # loop again because there is more than one coordinates in one line
        for n in range(len(i[1:])): # range does not include row number
            i[n+1] = i[n+1].split(",")
            i[n+1] = list(map(int, i[n+1]))
            point_list += [i[n+1]]
        # print (point_list) # e.g.[[62, 31], [61, 137]]
        # add all coordinates of points to gt_point_list
        gt_point_list += [(point_list)]

# read test images in a file
test_data = os.listdir("TestImages")
# loop through each image
for n in test_data:
    # extract only number of test image
    num = n.replace("test-", "")
    num = num.replace(".pgm", "")
    num = int(num)

    # make a list for Point objects
    point_ob = []
    # change point data to Point objects by Point method of method.py
    for m in (gt_point_list[num]):
        x = m[1]
        y = m[0]
        point_ob += [cw.Point(x,y)]

    # execute a function for drawing ground truth box in an image
    gt = cw.Ground_Truth('TestImages/'+n, point_ob)
    # print (gt.Numgt_box) # get the number of ground-truth box
    # print (gt.coords) # get the top-left and bottom-right coordinates of the box
    # gt.draw_box # draw the box in the image

    # carry out detector in the image
    pr = cw.Prediction('TestImages/'+n,"hog_svm") # must declare which method you use (hog_svm or harr or lbp)
    # print (pr.Numdet_box)
    # print (pr.coords)
    # pr.draw_box

    # calculate the IoU and count the number of correct positive and false positive
    u = cw.Accuracy('TestImages/'+n, gt, pr)
    # input the IoU rates in a list
    accuracy =  (",".join(map(str,u.IoU_rate)))
    output = n,gt.Numgt_box,pr.Numdet_box,accuracy
    # print (output) # e.g. ('test-125.pgm', 1, 1, '6.4')

    # calculate the number of correct detection
    if  u.IoU_rate: # if the IoU rate is calculated
        accuracy = accuracy.split(",")
        accuracy_avg += ([float(i) for i in accuracy])
        # if the number of ground-truth box is equal to or more than the one of prediction,
        # plus the number of prediction
        if gt.Numgt_box == pr.Numdet_box or (gt.Numgt_box > pr.Numdet_box):
            correct_pos += pr.Numdet_box
        # if the number of prediction is over the one of ground-truth boxes,
        # plus the number of ground-truth box
        elif gt.Numgt_box < pr.Numdet_box:
            correct_pos += gt.Numgt_box


    # calculate the number of false positive detection
    # if IoU rate is null or the number of prediction does not match the one of ground-truth boxes
    if not u.IoU_rate or not(gt.Numgt_box == pr.Numdet_box):
        # print (output)
        # if IoU rate is null, plus the number of prediction
        if not u.IoU_rate:
            false_pos += pr.Numdet_box
        # if the number of prediction is over the one of ground-truth boxes,
        # add the difference
        elif gt.Numgt_box < pr.Numdet_box:
            false_pos += pr.Numdet_box - gt.Numgt_box

print (correct_pos)
print (false_pos)
# print (np.mean(accuracy_avg))

