# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:47:12 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow
import cv2
import matplotlib.pyplot as plt


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.load1.clicked.connect(self.buttonClicked_load1)
        self.ui.load2.clicked.connect(self.buttonClicked_load2)
        self.ui.keypoints.clicked.connect(lambda : Q4_1(self.img1))
        self.ui.matchedKeypoints.clicked.connect(lambda : Q4_2(self.img1,self.img2))
    
    def buttonClicked_load1(self):
        self.img1, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.img1)
    
    def buttonClicked_load2(self):
        self.img2, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.img2)

def Q4_1(img1):
    image1 = cv2.imread(img1)  
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    #keypoints
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    imgpoint = cv2.drawKeypoints(gray1,keypoints_1,image1, (0,255,0))
    cv2.imshow('img1',imgpoint)
    
def Q4_2(img1,img2):
    image1 = cv2.imread(img1)  
    image2 = cv2.imread(img2)
    
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)
    
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    match = bf.match(descriptors_1,descriptors_2)
    match = sorted(match, key = lambda x:x.distance)
    
    imgpoint = cv2.drawKeypoints(image1,keypoints_1,image1, (0,255,0))
    showmatch = cv2.drawMatches(imgpoint, keypoints_1, image2, keypoints_2, match[:50], image2, flags=2)
    plt.imshow(showmatch)
    plt.show()  
    
    #matchesMask = [[1, 0] for i in range(len(match))]
    #for i, (m, n) in enumerate(match):
       # if m.distance < 0.4*n.distance:
            #matchesMask[i] = [1, 0]
    #drawpara = dict(singlePointColor=(0, 255, 0), matchColor=(255, 0, 0), matchesMask=matchesMask, flags=2)
    #image3 = cv.drawMatchesKnn(image1, keypoints_1, image2, keypoints_2, match, None, **drawpara)
    #cv.imshow("flann_match_demo", image3)
