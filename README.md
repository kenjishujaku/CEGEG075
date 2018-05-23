# CEGEG075：Image Understanding Coursework
This coursework aims to detect a car or cars in an image, using OpenCV package of Python. Sample images are downloaded from UIUC Image Database for Car Detection (http://cogcomp.org/Data/Car/).

What the programming files do is the following:
  1. train detectors on a number of sample images with machine learning algorithms
  2. detect specific objects in an image
  3. evaluate the accuracy (Intersection over Union)
  
The detection is like below images:(Green Box: Ground-Truth Box, Red Box: Detection)

<img src="https://user-images.githubusercontent.com/39371515/40451676-4a7c957e-5ed7-11e8-8161-39db4e15a5ae.png" width="430"><img src="https://user-images.githubusercontent.com/39371515/40451685-51146f2e-5ed7-11e8-80b4-91dac32a32e5.png" width="430">


There are mainly two files, scripts and xml_files. Each file is described below.
<h4> scripts </h4>
1. train_HoG_SVM.py <br />
This is to train SVM classifier with HoG features in an image. To implement this, you need to create filies for storing positive samples and negative samples respectively. The trained SVM classifier will be sotred as xml file in a specific folder.<br />
2. method.py <br />
This is to combine multiple functions needed to carry out object detection. You have to specify your own file pathes of created xml files of classifiers.<br />
3. implement.py <br />
This is to perform object detection in test images and calculate overlaying rate of detection to ground truth box. The number of positive detection and false positive detection are also counted. 

<h4> xml files </h4>
These xml files are detectors trained with specific features such as Harr-Like, HoG and LBP using opencv function. At detail, go to opencv document webpage (https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html). These are used in method.py and implement.py.

<h3> Usage and Notes </h3>
Download these files and scripts, and excute on python. <br />
Make sure that sample images are stored in files and adjust the path names in a script<br />
There is no guarantee that this will work perfectly <br />
Noted that some versions of python do not support SVM related functions and LBP classifier
