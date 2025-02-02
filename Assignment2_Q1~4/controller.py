# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:47:12 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow
import cv2
import copy
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
import glob
import os
import random
import sys
import numpy as np
import tensorflow as tf
import cv2.aruco as aruco
import argparse
import imutils
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn import decomposition

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.imgs = []

    def setup_control(self):
        self.ui.loadVideo.clicked.connect(self.buttonClicked_loadVideo)
        self.ui.loadImage.clicked.connect(self.buttonClicked_loadImage)
        self.ui.loadFolder.clicked.connect(self.buttonClicked_loadFolder)
        self.ui.backGround.clicked.connect(self.Q1)
        self.ui.preProcess.clicked.connect(self.Q2_1)
        self.ui.videoTrack.clicked.connect(self.Q2_2)
        self.ui.perPective.clicked.connect(self.Q3)
        self.ui.comTheRecon.clicked.connect(self.Q4_2)
        self.ui.imgRecon.clicked.connect(self.Q4_1)
        
    
    def buttonClicked_loadVideo(self):
        self.video, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.video)
        self.ui.labelVideo.setText(self.video)
        
    
    def buttonClicked_loadImage(self):
        self.image, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.image)
        self.ui.labelImage.setText(self.image)
        
    def buttonClicked_loadFolder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Open folder" ,"./")
        print(self.folder_path)
        self.ui.labelFolder.setText(self.folder_path)
    def Q1(self):
        cap = cv2.VideoCapture(self.video)
        images = []
        while True:
            retval, frame = cap.read()
            if retval == False:
                break
            
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            images.append(frame_gray)
            # removing the images after every 60 image
            if len(images)==60:
                images.pop(0)
            
            image = np.array(images)
            image = np.mean(image,axis=0)
            image = image.astype(np.uint8)

            # foreground will be background - curr frame
            foreground_image = cv2.absdiff(frame_gray,image)
            a = np.array([0],np.uint8)
            b = np.array([255],np.uint8)


            img = np.where(foreground_image>70,b,a)
            
            output = cv2.bitwise_and(frame,frame, mask=img)

            #cv2.imshow('frame', frame)
            cv2.imshow('mask',img)
            #cv2.imshow('combine', output)
            result = np.hstack([frame,output])
            cv2.imshow('result', result)
            if cv2.waitKey(24) & 0xFF == 27:
                break
                
        cap.release()           
        cv2.destroyAllWindows()
        
    def Q2_1(self):
        cap = cv2.VideoCapture(self.video)
        _, first_frame = cap.read()
        first_frame = cv2.convertScaleAbs(first_frame)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.84
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(first_frame)

        keyP = cv2.KeyPoint_convert(keypoints)
        for x in range(len(keyP)):
            (P_Y, P_X) = keyP[x]
            # print((P_Y, P_X))
            im_rec = cv2.rectangle(first_frame, (int(P_Y - 5), int(P_X - 5)), (int(P_Y + 5), int(P_X + 5)), (0, 0, 255), 1)
            im_rec = cv2.line(im_rec,(int(P_Y - 5),int(P_X)),(int(P_Y + 5),int(P_X)), (0, 0, 255), 1)
            im_rec = cv2.line(im_rec,(int(P_Y),int(P_X - 5)),(int(P_Y),int(P_X + 5)), (0, 0, 255), 1)
        cv2.imshow("Circle detect", im_rec)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

    def Q2_2(self):
        lk_params = dict(winSize=(21, 21),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        cap = cv2.VideoCapture(self.video)
        ret, old_frame = cap.read()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.84
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(old_frame)

        KeyP = np.reshape(cv2.KeyPoint_convert(keypoints), (-1, 1, 2))
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while (1):
            ret, frame = cap.read()

            if not ret:
                break
            im_rec = frame.copy()
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, KeyP, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = KeyP[st == 1]
            
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
                im_rec = cv2.circle(im_rec,(int(a),int(b)),5,(0, 255, 255), -1)

            imCombine = cv2.add(im_rec, mask)
            if ret == True:
                cv2.imshow('point', imCombine)
                old_frame = frame.copy()
                KeyP = good_new.reshape(-1, 1, 2)
            else:
                cap.release()
                break

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def Q3(self):
        video=cv2.VideoCapture(self.video)
        image=cv2.imread(self.image)
        ret , frame =video.read()
        grayFrame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        dic = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters =  cv2.aruco.DetectorParameters_create()
        corners , ids , reject =  cv2.aruco.detectMarkers(grayFrame , dic , parameters = parameters)
        ids = ids.squeeze()              
        while (video.isOpened()):
            try:
                ret, frame = video.read()

                if frame is None: break
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters_create()
                markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(
                    frame, dictionary, parameters=parameters)

                index = np.squeeze(np.where(markerIDs == ids[2]))
                refPt1 = np.squeeze(markerCorners[index[0]])[0]

                index = np.squeeze(np.where(markerIDs == ids[1]))
                refPt2 = np.squeeze(markerCorners[index[0]])[1]

                distance = np.linalg.norm(refPt1 - refPt2)

                scalingFac = 0.02
                pts_dst = [
                    [refPt1[0] - round(scalingFac * distance), refPt1[1] - round((scalingFac * distance))]]
                pts_dst = pts_dst + \
                          [[refPt2[0] + round(scalingFac * distance),
                            refPt2[1] - round(scalingFac * distance)]]

                index = np.squeeze(np.where(markerIDs == ids[0]))
                refPt3 = np.squeeze(markerCorners[index[0]])[2]
                pts_dst = pts_dst + \
                          [[refPt3[0] + round(scalingFac * distance),
                            refPt3[1] + round(scalingFac * distance)]]

                index = np.squeeze(np.where(markerIDs == ids[3]))
                refPt4 = np.squeeze(markerCorners[index[0]])[3]
                pts_dst = pts_dst + \
                          [[refPt4[0] - round(scalingFac * distance),
                            refPt4[1] + round(scalingFac * distance)]]
                pts_src = [[0, 0], [image.shape[1], 0], [
                    image.shape[1], image.shape[0]], [0, image.shape[0]]]

                pts_dst = np.float32(pts_dst)
                pts_src = np.float32(pts_src)

                h, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
                im_out = cv2.warpPerspective(image, h, (frame.shape[1], frame.shape[0]))

                res = np.where(im_out == 0, frame, im_out)
                res = cv2.resize(res, (640, 480), interpolation=cv2.INTER_CUBIC)
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                result = np.hstack([frame, res])
                cv2.imshow("result", result)
                if (cv2.waitKey(30) & 0xff == ord('q')):
                    break
            except:
                pass
        video.release()           
        cv2.destroyAllWindows()

    def Q4_1(self):
        def pca_reconstruction(image, num_features=33):
            """
            This function is equivalent to:
            from sklearn.decomposition import PCA
            pca = PCA(num_features)
            recon = pca.fit_transform(image)
            recon = pca.inverse_transform(recon)
            """
            average_image = np.expand_dims(np.mean(image, axis=1), axis=1)
            X = image - average_image
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            recon = average_image + np.matmul(np.matmul(U[:, :num_features], U[:, :num_features].T), X)
            return np.uint8(np.absolute(recon))

        np.random.seed(42)
        img_path = sorted(glob.glob(self.folder_path + "/*.jpg"))
        gray_img = None
        original_img = None

        for img_test in img_path:
            if gray_img is None:
                img = cv2.imread(img_test)
                original_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
            else:
                img = cv2.imread(img_test)
                img1 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                img2 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
                original_img = np.concatenate((original_img, img1), axis=0)
                gray_img = np.concatenate((gray_img, img2), axis=0)

        img_shape = original_img.shape
        r = original_img[:, :, :, 0].reshape(img_shape[0], -1)
        g = original_img[:, :, :, 1].reshape(img_shape[0], -1)
        b = original_img[:, :, :, 2].reshape(img_shape[0], -1)
        r_r, r_g, r_b = pca_reconstruction(r), pca_reconstruction(g), pca_reconstruction(b)
        recon_img = np.dstack((r_r, r_g, r_b))
        recon_img = np.reshape(recon_img, img_shape)

        # Setup a figure 6 inches by 6 inches
        fig = plt.figure(figsize=(15, 4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        length = 15
        for i in range(length):
            ax1 = fig.add_subplot(4, length, i + 1, xticks=[], yticks=[])
            ax1.imshow(original_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax2 = fig.add_subplot(4, length, i + length + 1, xticks=[], yticks=[])
            ax2.imshow(recon_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax3 = fig.add_subplot(4, length, i + length * 2 + 1, xticks=[], yticks=[])
            ax3.imshow(original_img[i + length], cmap=plt.cm.bone, interpolation='nearest')
            ax4 = fig.add_subplot(4, length, i + length * 3 + 1, xticks=[], yticks=[])
            ax4.imshow(recon_img[i + length], cmap=plt.cm.bone, interpolation='nearest')
        plt.show()
 
        
    def Q4_2(self):
        def pca_reconstruction(image, num_features=33):
            """
            This function is equivalent to:
            from sklearn.decomposition import PCA
            pca = PCA(num_features)
            recon = pca.fit_transform(image)
            recon = pca.inverse_transform(recon)
            """
            average_image = np.expand_dims(np.mean(image, axis=1), axis=1)
            X = image - average_image
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            recon = average_image + np.matmul(np.matmul(U[:, :num_features], U[:, :num_features].T), X)
            return np.uint8(np.absolute(recon))

        def reconstruction_error(gray):
            gray = gray.reshape(gray.shape[0], -1)
            gray_recon = pca_reconstruction(gray)
            re = np.sum(np.abs(gray - gray_recon), axis=1)
            return np.mean(re)
        np.random.seed(42)
        gray_img = None
        original_img = None
        for i in range(1, 31):
            if gray_img is None:
                img = cv2.imread(self.folder_path + '/sample (' + str(i) + ').jpg')
                original_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
            else:
                img = cv2.imread(self.folder_path + '/sample (' + str(i) + ').jpg')
                img1 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                img2 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
                original_img = np.concatenate((original_img, img1), axis=0)
                gray_img = np.concatenate((gray_img, img2), axis=0)

        total_error = []
        for img in gray_img:
            error = reconstruction_error(img)
            total_error.append(error)
        print("Reconstruction Error:", total_error)