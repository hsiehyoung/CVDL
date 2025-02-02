# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(219, 307)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 191, 241))
        self.groupBox.setObjectName("groupBox")
        self.load1 = QtWidgets.QPushButton(self.groupBox)
        self.load1.setGeometry(QtCore.QRect(30, 30, 131, 31))
        self.load1.setObjectName("load1")
        self.load2 = QtWidgets.QPushButton(self.groupBox)
        self.load2.setGeometry(QtCore.QRect(30, 80, 131, 31))
        self.load2.setObjectName("load2")
        self.keypoints = QtWidgets.QPushButton(self.groupBox)
        self.keypoints.setGeometry(QtCore.QRect(30, 130, 131, 31))
        self.keypoints.setObjectName("keypoints")
        self.matchedKeypoints = QtWidgets.QPushButton(self.groupBox)
        self.matchedKeypoints.setGeometry(QtCore.QRect(30, 180, 131, 31))
        self.matchedKeypoints.setObjectName("matchedKeypoints")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 219, 21))
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
        self.groupBox.setTitle(_translate("MainWindow", "4.SIFT"))
        self.load1.setText(_translate("MainWindow", "Load Image 1"))
        self.load2.setText(_translate("MainWindow", "Load Image 2"))
        self.keypoints.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.matchedKeypoints.setText(_translate("MainWindow", "4.2 Matched Keypoints"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

