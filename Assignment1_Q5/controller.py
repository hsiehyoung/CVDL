# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:47:12 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow
import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.compat.v1 as tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from torchsummary import summary
from torchvision import models

from torchvision import transforms
#from matplotlib import pyplot as plt, transforms

#from keras.models import Sequential, load_model
from keras.datasets import cifar10
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.utils import np_utils,plot_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#from keras.utils import np_utils,plot_model
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Model


hello=0
print("資料載入中，請燒等...")
(X_train, Y_train), (X_test,Y_test)  = cifar10.load_data()
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
labels_withindex = {0: 'airplane',1: 'automobile',2: 'bird',3: 'cat',4: 'deer',5: 'dog',6: 'frog',7: 'horse',8: 'ship',9: 'truck',}

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(X_train, Y_train,test_size=0.2)
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train,num_classes=10)
y_val=to_categorical(y_val,num_classes=10)
y_test=to_categorical(Y_test,num_classes=10)
x_train = x_train/224.0
x_val = x_val/224.0
X_test = X_test/224.0
print(x_train.shape,x_val.shape,X_test.shape)
print(y_train.shape,y_val.shape,y_test.shape)
train_datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,shear_range = 0.1, horizontal_flip=True, vertical_flip=False)
train_datagen.fit(x_train)
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.6, min_lr=0.00001)

def modeling1(vgg_model):

    model=tf.keras.models.Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(.25))
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(.25))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dense(10,activation = 'softmax'))

    return model
    

def modeling():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    x_train = X_train.astype('float32')/255
    x_test = X_test.astype('float32')/255
    y_train = np_utils.to_categorical(Y_train)
    y_test = np_utils.to_categorical(Y_test)

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    result = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test), shuffle=True, verbose=1)
    model.save('CIFAR10_model_augmentation.h5')
    return result


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.load.clicked.connect(self.buttonClicked_load)
        self.ui.showTrain.clicked.connect(self.Q5_1)
        self.ui.showModel.clicked.connect(self.Q5_2)
        self.ui.showData.clicked.connect(self.Q5_3)
        self.ui.showAccuracy.clicked.connect(self.Q5_4)
        self.ui.inference.clicked.connect(self.Q5_5)
    
    def buttonClicked_load(self):
        self.img, fileType = QtWidgets.QFileDialog.getOpenFileName(self,'open file','./')
        print(self.img)
        scene = QtWidgets.QGraphicsScene()
        scene.setSceneRect(50, 50, 100, 100)
        showimg = QtGui.QPixmap(self.img)
        scene.addPixmap(showimg)
        self.ui.show.setScene(scene)



    
    def Q5_1(self):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8))
        index = random.randint(1, 50000)
        for i in range(3):
            for j in range(3):
                axes[i, j].set_title(labels[Y_train[index][0]])
                axes[i, j].imshow(X_train[index])
                axes[i, j].get_xaxis().set_visible(False)
                axes[i, j].get_yaxis().set_visible(False)
                index += 1
        plt.show()
        
    def Q5_2(self):
        vgg_model = tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_shape=(32,32,3))
        model=modeling1(vgg_model)
        model= models.vgg19()
        summary(model, (3, 224, 224)) 
    
    def Q5_3(self):
        img= self.img;
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding = 15, padding_mode = 'edge')])
        qimg1 = transform1(img)
        qimg2 = transform1(img)
        qimg3 = transform1(img)
        qimg1.convert('RGB')
        qimg2.convert('RGB')
        qimg3.convert('RGB')

        plt.subplot(131)
        plt.imshow(qimg1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(132)
        plt.imshow(qimg2)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(133)
        plt.imshow(qimg3)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Q5_4(self):
        """
        vgg_model = tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_shape=(32,32,3))
        model=modeling(vgg_model)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(
            optimizer = optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
        result=model.fit(
            train_datagen.flow(x_train, y_train, batch_size = 64),
            validation_data = (x_val, y_val),
            epochs = 30,
            verbose = 1,
            callbacks = [learning_rate_reduction]
            )
        model.save('VGG19_CIFAR10_model.h5')
        """
        result= modeling()
        acc = result.history['accuracy']
        val_acc = result.history['val_accuracy']
        loss = result.history['loss']
        val_loss = result.history['val_loss']
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title("Accuracy")
        plt.plot(acc,color = 'blue',label = 'Training')
        plt.plot(val_acc,color = 'orange',label = 'Testing')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.subplot(1, 2, 2)
        plt.title('Loss')
        plt.plot(loss,color = 'blue',label = 'Training')
        plt.plot(val_loss,color = 'orange',label = 'Validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()
    
    def Q5_5(self):
        # load the image
        img = self.img
        # load model
        model = load_model('CIFAR10_model_augmentation.h5')
        # predict the class
        result = model.predict(img) 
        result1 = np.max(result)
        result2 = np.argmax(result,axis=1)
        prediction_label = labels_withindex[result2[0]]
        msg = 'Confidence = ' + str(result1) + '\n' +'Prediction Label:'+ str(prediction_label)
        self.ui.label.setText(msg)
        # entry point, run the example


    