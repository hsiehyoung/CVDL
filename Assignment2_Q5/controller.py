# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:47:12 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from UI import Ui_MainWindow
import sys
import numpy as np
import cv2
import math
import glob
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt    
import tensorflow.keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Flatten, Activation,Conv2D, MaxPooling2D
import datetime
# import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
import os
import random
import cv2
import tensorflow_addons as tfa







class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        

    def setup_control(self):
        self.ui.loadImage.clicked.connect(self.buttonClicked_loadImage)
        self.ui.showImage.clicked.connect(self.Q1)
        self.ui.showDist.clicked.connect(self.Q2)
        self.ui.showModel.clicked.connect(self.Q3)
        self.ui.showComp.clicked.connect(self.Q4)
        self.ui.inference.clicked.connect(self.Q5)
        
    
    def buttonClicked_loadImage(self):
        self.image, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.image)
        self.img = cv2.imread(self.image)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2RGB)
        height, width, channel = self.img.shape
        bytesPerline=channel*width
        qImg = QImage(self.img, width,height, bytesPerline, QImage.Format_RGB888)
        #resize_img = cv2.resize(self.image,(224,224))
        #scene = QtWidgets.QGraphicsScene()
        #scene.setSceneRect(50, 50, 100, 100)
        #showimg = QtGui.QPixmap(self.image)
        #scene.addPixmap(showimg)
        #self.ui.resultShow.setScene(qImg)
        self.ui.showLabel.setPixmap(QPixmap.fromImage(qImg))
        self.ui.showLabel.setScaledContents(True)
        self.ui.resultLabel.setText("")
        
    def Q2(self):
        cv2.imshow("5.2",cv2.imread("5.2.png"))
        #cat_path = glob.glob('C:\\Users\\TSUNGYU\\Desktop\\Dataset_OpenCvDl_Hw2_Q5training_dataset\\resize\\Cat'+'/*.jpg' )
        #dog_path = glob.glob('C:\\Users\\TSUNGYU\\Desktop\\Dataset_OpenCvDl_Hw2_Q5training_dataset\\resize\\Dog'+'/*.jpg')
        #number = [len(cat_path), len(dog_path)]
        #animal = ['Cat','Dog']
        #x=np.arange(len(animal))
        #plt.bar(x, number)
        #plt.xticks(x,animal)
        #plt.title('Class Distribution')
        #plt.ylabel('Number of images')
        #ax = plt.gca()
        #plt.bar_label(ax.containers[0])
        #plt.savefig('5.2.png')
        #plt.show()
        
        
    def Q4(self):
        cv2.imshow("5.4",cv2.imread("5.4.png"))
        #ds = tf.keras.utils.image_dataset_from_directory("C:\\Users\\TSUNGYU\\Desktop\\Dataset_OpenCvDl_Hw2_Q5validation_dataset", labels = "inferred", image_size=(224, 224))
        #model_binary = tf.keras.models.load_model("inference/Q5_Model_binary.h5")
        #acc_binary = model_binary.evaluate(ds,return_dict = True)['accuracy']
        #model_fl = tf.keras.models.load_model("inference/Q5_Model_focal.h5")
        #acc_fl = model_fl.evaluate(ds,return_dict = True)['accuracy']
        #x = ["Binary Cross Entropy", "Focal Loss"]
        #y = [acc_binary*100,acc_fl*100]
        #plt.figure("5.4")
        #plt.title('Accuracy Comparison')
        #plt.bar(x,y)
        #plt.ylabel("Accuracy(%)")
        #ax=plt.gca()
        #plt.bar_label(ax.containers[0])
        #plt.savefig("Q5.4.png")
        #plt.show()
        
    def Q3(self):
        model_binary = tf.keras.models.load_model("inference/Q5_Model_binary.h5")
        print("Binary Cross Entropy")
        tf.keras.Model.summary(model_binary)
        model_binary = tf.keras.models.load_model("inference/Q5_Model_focal.h5")
        print("Focal Loss")
        tf.keras.Model.summary(model_binary)
        
    def Q1(self):
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 8))
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        show_it = datagen.flow_from_directory("inference_dataset",class_mode='binary', batch_size=1, target_size=(224, 224))
        number = random.randint(1,10)
        cat = 0
        dog = 0
        while(cat==0 or dog==0):
            number = random.randint(1,10)
            if show_it[number][1] == 1:
                axes[0].set_title("Dog")
                axes[0].imshow(show_it[number][0][0,:,:,:])
                axes[0].get_xaxis().set_visible(False)
                axes[0].get_yaxis().set_visible(False)
                dog=1
            if show_it[number][1] == 0:
                axes[1].set_title("Cat")
                axes[1].imshow(show_it[number][0][0,:,:,:])
                axes[1].get_xaxis().set_visible(False)
                axes[1].get_yaxis().set_visible(False)
                cat=1
        #plt.imshow(show_it[number][0][0,:,:,:])
        plt.show()
        
    def Q5(self):
        #print(self.image)
        model = tf.keras.models.load_model("inference/Q5_Model_binary.h5")
        resize_img = cv2.resize(self.img,(224,224))
        result =np.array([resize_img])
        pred = model.predict(result)
        if(pred[0][0]<0.5):
            self.ui.resultLabel.setText("prediction:Cat")
        else:
            self.ui.resultLabel.setText("prediction:Dog")