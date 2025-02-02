from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow

import cv2
import sys
import math
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import os 
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog

global folder_path


def Q1_1(folder_path):
    global mtx
    global rvecs
    global tvecs
    global dist
    global Homo
    
    dirs = os.listdir(folder_path)
    is_file = [f for f in dirs if os.path.isfile(os.path.join(folder_path,f))]
    count = 0
    for num in is_file:
        count=count+1
    print(count)
    print(folder_path)
    print(is_file)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    
    for i in range(0,count):
        filename = folder_path + '/'+ is_file[i]
        print(filename)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            cv2.namedWindow('window', 0)
            cv2.resizeWindow('window', 800,800)
            cv2.imshow('window', img)
            cv2.waitKey(500)
            
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    Homo, NOU = cv2.findHomography(corners, corners2)
    
    
def Q1_2():
    global mtx
    print('Intrinsic:')
    print(mtx)
    
def Q1_3(folder_path,num):
    global rvecs
    global tvecs
    global Homo
    global rvecs_3d
    rvecs_3d = []
    dirs = os.listdir(folder_path)
    is_file = [f for f in dirs if os.path.isfile(os.path.join(folder_path,f))]
    count = 0
    for i in is_file:
        count=count+1
    for i in range(0,count):
        temp_rvecs_3d, a = cv2.Rodrigues(rvecs[i])
        rvecs_3d.append(temp_rvecs_3d)
    Extrinsic_mtx = np.concatenate((rvecs_3d[num-1], tvecs[num-1]), axis=1)
    print('Extrinsic:')
    print(Extrinsic_mtx)

def Q1_4():
    global dist
    print('Distortion:')
    print(dist)

def Q1_5(folder_path):
    global mtx
    global rvecs
    global tvecs
    global dist
    global Homo
    
    dirs = os.listdir(folder_path)
    is_file = [f for f in dirs if os.path.isfile(os.path.join(folder_path,f))]
    count = 0
    for num in is_file:
        count=count+1

    for i in range(0,count):
        filename = folder_path + '/'+ is_file[i]
        img = cv2.imread(filename)
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.namedWindow('Distorted', 0)
        cv2.resizeWindow('Distorted', 800,800)
        cv2.namedWindow('Undistorted', 0)
        cv2.resizeWindow('Undistorted', 800,800)
        cv2.imshow('Distorted', img)
        cv2.imshow('Undistorted',dst)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

def Q2_1(folder_path,text):
    
    fs = cv2.FileStorage('alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
    text = text.upper()
    input = list(text)
    #print(input)
    ch = []
    for word in input:
        ch.append(fs.getNode(word).mat())
    
    point1 =  ch[0].reshape(-1,3)
    for i in range(len(point1)):
        point1[i]= point1[i]+[7,5,0]
    point2 =  ch[1].reshape(-1,3)
    for i in range(len(point2)):
        point2[i]= point2[i]+[4,5,0]
    point3 =  ch[2].reshape(-1,3)
    for i in range(len(point3)):
        point3[i]= point3[i]+[1,5,0]
    point4 =  ch[3].reshape(-1,3)
    for i in range(len(point4)):
        point4[i]= point4[i]+[7,2,0]
    point5 =  ch[4].reshape(-1,3)
    for i in range(len(point5)):
        point5[i]= point5[i]+[4,2,0]
    point6 =  ch[5].reshape(-1,3)
    for i in range(len(point6)):
        point6[i]= point6[i]+[1,2,0]
    #ch = fs.getNode('K').mat()
    #print (ch)
    #print(point1)
    point1 = np.float32(point1)
    point2 = np.float32(point2)
    point3 = np.float32(point3)
    point4 = np.float32(point4)
    point5 = np.float32(point5)
    point6 = np.float32(point6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    dirs = os.listdir(folder_path)
    is_file = [f for f in dirs if os.path.isfile(os.path.join(folder_path,f))]
    count = 0
    for num in is_file:
        count=count+1

    for i in range(0,count):
        filename = folder_path + '/'+ is_file[i]
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        points_3D = []
        points_2D = []
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            points_3D.append(objp)
            points_2D.append(corners)

            _, mtx, dist, _, _ = cv2.calibrateCamera(points_3D, points_2D, gray.shape[::-1], None, None)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 8), (-1, -1), criteria)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            imagePoints, jocobian = cv2.projectPoints(point1, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point1)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point2, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point2)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point3, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point3)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point4, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point4)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point5, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point5)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point6, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point6)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            cv2.namedWindow('image', 0)
            cv2.resizeWindow('image',800,800)
            cv2.imshow("image", img)
            cv2.waitKey(500)

def Q2_2(folder_path,text):
    fs = cv2.FileStorage('alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
    text = text.upper()
    input = list(text)
    #print(input)
    ch = []
    for word in input:
        ch.append(fs.getNode(word).mat())
    print(ch[0])
    
    point1 =  ch[0].reshape(-1,3)
    for i in range(len(point1)):
        point1[i]= point1[i]+[7,5,0]
    point2 =  ch[1].reshape(-1,3)
    for i in range(len(point2)):
        point2[i]= point2[i]+[4,5,0]
    point3 =  ch[2].reshape(-1,3)
    for i in range(len(point3)):
        point3[i]= point3[i]+[1,5,0]
    point4 =  ch[3].reshape(-1,3)
    for i in range(len(point4)):
        point4[i]= point4[i]+[7,2,0]
    point5 =  ch[4].reshape(-1,3)
    for i in range(len(point5)):
        point5[i]= point5[i]+[4,2,0]
    point6 =  ch[5].reshape(-1,3)
    for i in range(len(point6)):
        point6[i]= point6[i]+[1,2,0]
    #ch = fs.getNode('K').mat()
    #print (ch)
    #print(point1)
    point1 = np.float32(point1)
    point2 = np.float32(point2)
    point3 = np.float32(point3)
    point4 = np.float32(point4)
    point5 = np.float32(point5)
    point6 = np.float32(point6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    dirs = os.listdir(folder_path)
    is_file = [f for f in dirs if os.path.isfile(os.path.join(folder_path,f))]
    count = 0
    for num in is_file:
        count=count+1

    for i in range(0,count):
        filename = folder_path + '/'+ is_file[i]
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        points_3D = []
        points_2D = []
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            points_3D.append(objp)
            points_2D.append(corners)

            _, mtx, dist, _, _ = cv2.calibrateCamera(points_3D, points_2D, gray.shape[::-1], None, None)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 8), (-1, -1), criteria)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            imagePoints, jocobian = cv2.projectPoints(point1, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point1)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point2, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point2)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point3, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point3)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point4, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point4)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point5, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point5)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            imagePoints, jocobian = cv2.projectPoints(point6, rvecs, tvecs, mtx, dist)
            imagePoints = np.int32(imagePoints).reshape(-1, 2)
            for i in range(len(point6)):
                if i%2==0:
                    cv2.line(img, tuple(imagePoints[i]), tuple(imagePoints[i+1]), (0, 0, 255), 3)
            
            cv2.namedWindow('image', 0)
            cv2.resizeWindow('image',800,800)
            cv2.imshow("image", img)
            cv2.waitKey(500)    
        
def Q3_1(imgLeft,imgRight):
    imgLC = cv2.imread(imgLeft)
    imgRC = cv2.imread(imgRight)
    stereo = cv2.StereoBM_create(numDisparities = 256,
                            blockSize = 25)
    imgL = cv2.imread(imgLeft, 0)
    imgR = cv2.imread(imgRight, 0)
    disparity = stereo.compute(imgL, imgR)

    disparity = stereo.compute(imgL, imgR)
    disparity -= 123
    disparity = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255)

    cv2.namedWindow('imgL', 0)
    cv2.imshow('imgL', imgLC)
    cv2.namedWindow('imgR', 0)
    cv2.imshow('imgR', imgRC)
    cv2.namedWindow('disparity', 0) 
    cv2.imshow('disparity', disparity)
    
    def mouse_handler(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data['imgR'], (x,y), 25, (124,252,0), -1)
            cv2.imshow('imgR', data['imgR'])
            print("get points: (x, y) = ({}, {})".format(x, y))
    def get_points(im):
        data = {}
        data['imgR'] = imgRC.copy()
        #data['points'] = []
        cv2.setMouseCallback("imgL", mouse_handler, data)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    plt.imshow(disparity, 'gray')  
    plt.axis('off')
    plt.show()
    



class MainWindow_controller(QtWidgets.QMainWindow):
    
    

    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.ui.findCorner.setEnabled(False)
        self.ui.findInst.setEnabled(False)
        self.ui.findExt.setEnabled(False)
        self.ui.findDist.setEnabled(False)
        self.ui.showResult.setEnabled(False)
        self.ui.showOnBoard.setEnabled(False)
        self.ui.showVertical.setEnabled(False)
        self.ui.stereoMap.setEnabled(False)

    def setup_control(self):
        # TODO
        # qpushbutton doc: https://doc.qt.io/qt-5/qpushbutton.html
        
        self.ui.loadFolder.clicked.connect(self.buttonClicked_loadFolder)
        self.ui.LoadL.clicked.connect(self.buttonClicked_loadL)
        self.ui.loadR.clicked.connect(self.buttonClicked_loadR)
        self.ui.findCorner.clicked.connect(lambda : Q1_1(self.folder_path))
        self.ui.findInst.clicked.connect(lambda : Q1_2())
        #self.ui.findExt.clicked.connect(self.buttonClicked_Q1_3)
        self.ui.findExt.clicked.connect(lambda : Q1_3(self.folder_path,self.ui.num.currentIndex()+1))
        self.ui.findDist.clicked.connect(lambda : Q1_4())
        self.ui.showResult.clicked.connect(lambda : Q1_5(self.folder_path))
        self.ui.showOnBoard.clicked.connect(lambda : Q2_1(self.folder_path,self.ui.text.toPlainText()))
        self.ui.showVertical.clicked.connect(lambda : Q2_2(self.folder_path,self.ui.text.toPlainText()))
        self.ui.stereoMap.clicked.connect(lambda : Q3_1(self.imgL,self.imgR))
    
    def buttonClicked_loadFolder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Open folder" ,"./")
        print(self.folder_path)
        dirs = os.listdir(self.folder_path)
        is_file = [f for f in dirs if os.path.isfile(os.path.join(self.folder_path,f))]
        print(is_file)
        count = 0
        for num in is_file:
            count=count+1
        print(count)

        self.ui.findCorner.setEnabled(True)
        self.ui.showOnBoard.setEnabled(True)
        self.ui.showVertical.setEnabled(True)
        self.ui.findInst.setEnabled(True)
        self.ui.findDist.setEnabled(True)
        self.ui.findExt.setEnabled(True)
        self.ui.showResult.setEnabled(True)

    def buttonClicked_loadL(self):
        self.imgL, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.imgL)


    def buttonClicked_loadR(self):
        self.imgR, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.imgR)
        self.ui.stereoMap.setEnabled(True)