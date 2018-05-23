# CEGEG075：Image Understanding Coursework
This coursework aims to detect a car or cars in an image, using OpenCV package of Python. Sample images are downloaded from UIUC Image Database for Car Detection(http://cogcomp.org/Data/Car/).

What the programming files do is the following:
  1. train detectors on a number of sample images with machine learning algorithms
  2. detect specific objects in an image
  3. evaluate the accuracy
  
The detection is like below images:

<img src="https://user-images.githubusercontent.com/39371515/40451676-4a7c957e-5ed7-11e8-8161-39db4e15a5ae.png" width="430"><img src="https://user-images.githubusercontent.com/39371515/40451685-51146f2e-5ed7-11e8-80b4-91dac32a32e5.png" width="430">
Green Box: Ground-Truth Box, Red Box: Detection

There are mainly two files, scripts and xml_files. Each file is described below.

# Scripts consist of three scripts 
1. train_HoG_SVM.py
This is to train SVM classifier with HoG features in an image. To implement this, you need to create filies for storing positive samples and negative samples respectively. 
The trained SVM classifier will be sotred as xml file in a specific folder

2. method.py
This is to combine multiple functions needed to carry out object detection. You have to specify your own file pathes of created xml files of classifiers

3. implement.py
This is to perform object detection in test images and calculate overlaying rate of detection to ground truth box. The number of positive detection and false positive detection are also counted. 

# xml files 
