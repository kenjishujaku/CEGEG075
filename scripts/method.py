##### Image Classifying Method #####

# import required libraries
import os
import cv2
import numpy as np
import pylab as plt
import imutils

# define parameters for HoG descriptor
win_size = (100,40)
block_size = (8,8)
block_stride = (4,4)
cell_size = (4,4)
bins = 9
numPos = 550
numNeg = 500
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# create Point object
class Point(object):
    # follow pixel coordinate system which is the origin is the top-left corner of an image
    # x: column, y: row
    def __init__(self,x=0.0,y=0.0):
        self.__point_coords = (x,y)
    # attribute for accessing point coordinates
    @property
    def coords(self):
        return self.__point_coords

# create Ground_Truth object for creating a ground truth box
class Ground_Truth(object):
    # point list must contain Point objects created from top-left and bottom-right coordinates
    # of given ground truth boxes
    def __init__(self,img,point_list=None):
        if point_list is None:
            point_list = []
        # make an empty list for ground truth boxes
        self.gt_box = []

        # set x,y coordinates of a ground-truth box
        for k in range(len(point_list)):
            x = point_list[k].coords[0]
            y = point_list[k].coords[1]
            self.gt_box += [[(x,y),(x+100,y+40)]] # ground-truth box should be 100 by 40 window size

        self.__gt_box_list = self.gt_box
        # read an input image
        self.__gt_img = cv2.imread(img)

    @property
    def Numgt_box(self):
        # return count of ground-truth boxes
        return len(self.gt_box)

    @property
    def coords(self):
        # return a list of coordinates of ground-truth boxes
        return self.__gt_box_list

    @property
    def draw_box(self):
        # draw the ground-truth boxes in the image
        for n in self.__gt_box_list:
            cv2.rectangle(self.__gt_img,n[0],n[1],(0,255,0),2)
        plt.figure(figsize=(10,20))
        plt.imshow(self.__gt_img)
        plt.show()

# create a class for executing a detector to an image
class Prediction(object):
    def __init__(self,img,method):
        # read a image
        self.img = cv2.imread(img)
        self.car = 0
        pr_box = []
        tmp_box = []

        # carry out Harr-AdaBoost detector
        if method == "harr":
            classifier = cv2.CascadeClassifier("/home/kenji03/Desktop/CEGEG075/Coursework/Classifier/classifier/Harr-Like/harr-AdaBoost.xml")
            self.Detector = classifier.detectMultiScale(self.img,1.1,1)
            for (x, y, w, h) in self.Detector:
                tmp_box += [(x,y,x+w, y+h)]
            # apply non-maximum supprsesion
            pick = nms(np.array(tmp_box), 0.3)
            for (x1, y1, x2, y2) in pick:
                self.car += 1 # if the detector find a car, plus one
                pr_box += [[(x1, y1), (x2, y2)]]
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw the detected boxes

        # carry out LBP-AdaBoost detector
        elif method == "lbp":
            classifier = cv2.CascadeClassifier("/home/kenji03/Desktop/CEGEG075/Coursework/Classifier/classifier/LBP/lbp-AdaBoost.xml")
            self.Detector = classifier.detectMultiScale(self.img,1.1,1)
            for (x, y, w, h) in self.Detector:
                tmp_box += [(x,y, x+w, y+h)]
            # apply non-maximum suppression
            pick = nms(np.array(tmp_box), 0.3)
            for (x1, y1, x2, y2) in pick:
                self.car += 1 # if the detector find a car, plus one
                pr_box += [[(x1,y1),(x2,y2)]]
                cv2.rectangle(self.img, (x1,y1), (x2,y2), (255, 0, 0), 2) # draw the detected boxes

        # carry out HoG-SVM detector
        elif method == "hog_svm":
            # load the created svm.xml
            svm = cv2.ml.SVM_load("/home/kenji03/Desktop/CEGEG075/Coursework/Classifier/classifier/HOG_SVM/HOG_SVM.xml")
            # get an input image's height and width
            height, width = self.img.shape[:2]
            # create zero array with same height and width as input image
            detect = np.zeros((height, width), dtype=np.int32)
            detect_boxes = []
            # loop over with resized images
            for resized in pyramid(self.img, scale=1.2):
                # loop over with sliding windows
                for (x, y, window) in sliding_window(resized, stepSize=5, windowSize=(100, 40)):
                    # if the resized image is smaller than the window size, ignore
                    if (x+100) > resized.shape[1] or (y+40) > resized.shape[0]:
                        continue
                    # take out the values from created zero array at a place as same as the region of a sliding window
                    detect_part = detect[y:y+40, x:x+100]
                    # extract HoG features
                    hog_feature = hog.compute(window)
                    # vectorize the features
                    hog_feature_reshaped = np.reshape(hog_feature, (1, 7776))
                    # apply svm classification
                    result = svm.predict(hog_feature_reshaped)

                    # if the result is positive
                    if result[1][0, 0] == 1:
                        # print("found a car!")
                        detect_part.fill(1) # change zero to one in the regions of sliding window in the image
                        detect_boxes += [(x,y,x+100,y+40)]

            # apply non-maximum suppression
            pick = nms(np.array(detect_boxes),0.3)
            for (x, y, w, h) in pick:
                self.car += 1 # if the detector find a car, plus one
                pr_box += [[(x,y),(w,h)]]
                cv2.rectangle(self.img, (x,y),(w,h), (255, 0, 0), 2) # draw the detected boxes

        self.__pr_box_list = pr_box

    @property
    def coords(self):
        # return the coordinates of prediction
        return self.__pr_box_list

    @property
    def Numdet_box(self):
        # return the number of detected boxes
        return self.car


    @property
    def draw_box(self):
        # draw the boxes
        plt.figure(figsize=(10,20))
        plt.imshow(self.img)
        plt.show()

# create class for calculating Intersect over Union rate
class Accuracy(Ground_Truth, Prediction):
    def __init__ (self, img,gt_object,pr_object):
        # read a input image
        self.__test_img = cv2.imread(img)
        # set variables for ground-truth object and prediction object
        self.__gt_object = gt_object
        self.__pr_object = pr_object
        self.__IoU_rate = []

        # loop over ground-truth objects
        for s in self.__gt_object.coords:
            cv2.rectangle(self.__test_img,s[0],s[1],(0,255,0),2) # draw the groung-truth box
            minIoU = float("-inf") # define the minimum IoU as -inf
            IoU_list = []

            # loop over detected boxes
            for t in self.__pr_object.coords:

                # set variables about ground-truth boxes
                GT_tl = s[0]
                GT_br = s[1]
                GT_width = 100
                GT_height = 40
                GT_centroid = (s[0][0]+50,s[0][1]+20) # define the centroid of ground-truth box
                # set variables about detected boxes
                PR_tl = t[0]
                PR_br = t[1]
                PR_width = abs(t[0][0]-t[1][0])
                PR_height = abs(t[0][1]-t[1][1])
                PR_centroid = (t[0][0]+PR_width/2,t[0][1]+PR_height) # define the centroid of detected boxes

                cv2.rectangle(self.__test_img, PR_tl, PR_br, (255, 0, 0), 2) # draw the detected boxes

                # set conditions whether detected boxes intersect with ground-truth boxes
                condition1 = abs(GT_centroid[0]-PR_centroid[0]) < abs(GT_width/2 + PR_width/2) # x direction
                condition2 = abs(GT_centroid[1]-PR_centroid[1]) < abs(GT_height/2 + PR_height/2) # y direction

                if condition1 and condition2: # if the conditions are met, calculate IoU rate
                    area_Union = abs(GT_br[0] - GT_tl[0]) * abs(GT_tl[1] - GT_br[1])
                    area_Intersection = abs(min(GT_br[0],PR_br[0])-max(GT_tl[0],PR_tl[0]))*abs(max(GT_tl[1],PR_tl[1])-min(GT_br[1],PR_br[1]))

                    IoU_rate = (float(area_Intersection) / 4000)*100

                    # if IoU rate is larger than the previous, update the value
                    # there are multiple detected boxes overlapping on a ground-truth box, thus
                    # choose the detected box that overlay at the largest area of the ground-truth box
                    if max(IoU_rate, minIoU)==IoU_rate:
                        minIoU = IoU_rate # update IoU rate
                        IoU_list += [(IoU_rate, t[0], t[1])] # including IoU rate and the coordinate of a detected box

            for i in IoU_list:
                if i[0] == minIoU:
                    self.__IoU_rate += [i[0]]
                    cv2.rectangle(self.__test_img,i[1],i[2], (255, 0, 0), 2) #draw the detected box

    @property
    def IoU_rate(self):
        # return IoU rate
        return self.__IoU_rate

    @property
    def draw_boxes(self):
        # draw the detected box
        plt.figure(figsize=(10,20))
        plt.imshow(self.__test_img)
        plt.show()

# create a function of resizing referred to
# <https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/>
def pyramid(img,scale=1.2,minSize=(100,40)):
    # yield the original image
    yield img
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(img.shape[1]/scale)
        img = imutils.resize(img,width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if img.shape[0] <= minSize[1] or img.shape[1] <= minSize[0]:
            break
        yield img

# create a function of sliding window referred to
# <https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/>
def sliding_window(img,stepSize,windowSize):
    for y in range(0,img.shape[0],stepSize):
        for x in range(0,img.shape[1],stepSize):
            yield (x,y,img[y:y+windowSize[1], x:x+windowSize[0]])


# create non maximum suppression function referred to
# <https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/>
def nms(boxes,overlap):
    if len(boxes)==0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of each detected box
    area = (x2-x1+1)*(y2-y1+1)
    # sort the y2 in ascending order
    idxs = np.argsort(y2)

    while len(idxs)>0:
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)

        # compute coordinates of a large rectangle
        xx1 = np.maximum(x1[i],x1[idxs[:last]])
        yy1 = np.minimum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        w = np.maximum(0,xx2-xx1+1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the overlapping area of the large rectangle with each detected box
        intersection = (w*h)/area[idxs[:last]]
        # remove a detected box overlapping that meets the threshold condition
        idxs = np.delete(idxs,np.concatenate(([last],np.where(intersection > overlap)[0])))

    return boxes[pick].astype("int")


