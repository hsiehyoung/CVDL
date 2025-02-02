# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 03:08:28 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from UI import Ui_MainWindow
import tensorflow as tf 
import matplotlib.pyplot as plt    
import tensorflow.keras


ds = tf.keras.utils.image_dataset_from_directory("C:\\Users\\TSUNGYU\\Desktop\\Dataset_OpenCvDl_Hw2_Q5validation_dataset", labels = "inferred", image_size=(224, 224))
model_binary = tf.keras.models.load_model("inference/Q5_Model_binary.h5")
acc_binary = model_binary.evaluate(ds,return_dict = True)['accuracy']
model_fl = tf.keras.models.load_model("inference/Q5_Model_focal.h5")
acc_fl = model_fl.evaluate(ds,return_dict = True)['accuracy']
x = ["Binary Cross Entropy", "Focal Loss"]
y = [acc_binary*100,acc_fl*100]
plt.figure("5.4")
plt.title('Accuracy Comparison')
plt.bar(x,y)
plt.ylabel("Accuracy(%)")
ax=plt.gca()
plt.bar_label(ax.containers[0])
plt.savefig("Q5.4.png")
plt.show()