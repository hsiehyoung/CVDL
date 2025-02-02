# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(334, 579)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelVideo = QtWidgets.QLabel(self.centralwidget)
        self.labelVideo.setGeometry(QtCore.QRect(60, 40, 231, 20))
        self.labelVideo.setObjectName("labelVideo")
        self.labelImage = QtWidgets.QLabel(self.centralwidget)
        self.labelImage.setGeometry(QtCore.QRect(60, 90, 211, 20))
        self.labelImage.setObjectName("labelImage")
        self.loadImage = QtWidgets.QPushButton(self.centralwidget)
        self.loadImage.setGeometry(QtCore.QRect(60, 70, 211, 23))
        self.loadImage.setObjectName("loadImage")
        self.labelFolder = QtWidgets.QLabel(self.centralwidget)
        self.labelFolder.setGeometry(QtCore.QRect(60, 140, 211, 20))
        self.labelFolder.setObjectName("labelFolder")
        self.loadFolder = QtWidgets.QPushButton(self.centralwidget)
        self.loadFolder.setGeometry(QtCore.QRect(60, 120, 211, 23))
        self.loadFolder.setObjectName("loadFolder")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 170, 271, 61))
        self.groupBox.setObjectName("groupBox")
        self.backGround = QtWidgets.QPushButton(self.groupBox)
        self.backGround.setGeometry(QtCore.QRect(30, 20, 211, 23))
        self.backGround.setObjectName("backGround")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 250, 271, 101))
        self.groupBox_2.setObjectName("groupBox_2")
        self.preProcess = QtWidgets.QPushButton(self.groupBox_2)
        self.preProcess.setGeometry(QtCore.QRect(30, 20, 211, 23))
        self.preProcess.setObjectName("preProcess")
        self.videoTrack = QtWidgets.QPushButton(self.groupBox_2)
        self.videoTrack.setGeometry(QtCore.QRect(30, 60, 211, 23))
        self.videoTrack.setObjectName("videoTrack")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 360, 271, 61))
        self.groupBox_3.setObjectName("groupBox_3")
        self.perPective = QtWidgets.QPushButton(self.groupBox_3)
        self.perPective.setGeometry(QtCore.QRect(30, 20, 211, 23))
        self.perPective.setObjectName("perPective")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(30, 430, 271, 101))
        self.groupBox_4.setObjectName("groupBox_4")
        self.imgRecon = QtWidgets.QPushButton(self.groupBox_4)
        self.imgRecon.setGeometry(QtCore.QRect(30, 20, 211, 23))
        self.imgRecon.setObjectName("imgRecon")
        self.comTheRecon = QtWidgets.QPushButton(self.groupBox_4)
        self.comTheRecon.setGeometry(QtCore.QRect(30, 60, 211, 23))
        self.comTheRecon.setObjectName("comTheRecon")
        self.loadVideo = QtWidgets.QPushButton(self.centralwidget)
        self.loadVideo.setGeometry(QtCore.QRect(60, 20, 211, 23))
        self.loadVideo.setObjectName("loadVideo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 334, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelVideo.setText(_translate("MainWindow", "No video loaded"))
        self.labelImage.setText(_translate("MainWindow", "No Image loaded"))
        self.loadImage.setText(_translate("MainWindow", "Load Image"))
        self.labelFolder.setText(_translate("MainWindow", "No Folder loaded"))
        self.loadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.groupBox.setTitle(_translate("MainWindow", "1.Background Subtraction"))
        self.backGround.setText(_translate("MainWindow", "1.1 Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.Optical Flow"))
        self.preProcess.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.videoTrack.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Perpective Transform"))
        self.perPective.setText(_translate("MainWindow", "3.1 Perpective Transform"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4.PCA"))
        self.imgRecon.setText(_translate("MainWindow", "4.1 ImageReconstruction"))
        self.comTheRecon.setText(_translate("MainWindow", "4.2 Compute the Reconstruction Error"))
        self.loadVideo.setText(_translate("MainWindow", "Load Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

