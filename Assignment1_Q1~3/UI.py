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
        MainWindow.resize(830, 346)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Load = QtWidgets.QGroupBox(self.centralwidget)
        self.Load.setGeometry(QtCore.QRect(30, 40, 161, 261))
        self.Load.setObjectName("Load")
        self.loadFolder = QtWidgets.QPushButton(self.Load)
        self.loadFolder.setGeometry(QtCore.QRect(20, 50, 121, 23))
        self.loadFolder.setObjectName("loadFolder")
        self.LoadL = QtWidgets.QPushButton(self.Load)
        self.LoadL.setGeometry(QtCore.QRect(20, 110, 121, 23))
        self.LoadL.setObjectName("LoadL")
        self.loadR = QtWidgets.QPushButton(self.Load)
        self.loadR.setGeometry(QtCore.QRect(20, 170, 121, 23))
        self.loadR.setObjectName("loadR")
        self.one = QtWidgets.QGroupBox(self.centralwidget)
        self.one.setGeometry(QtCore.QRect(230, 40, 161, 261))
        self.one.setObjectName("one")
        self.findCorner = QtWidgets.QPushButton(self.one)
        self.findCorner.setGeometry(QtCore.QRect(20, 30, 121, 23))
        self.findCorner.setObjectName("findCorner")
        self.groupBox = QtWidgets.QGroupBox(self.one)
        self.groupBox.setGeometry(QtCore.QRect(10, 100, 141, 80))
        self.groupBox.setObjectName("groupBox")
        self.num = QtWidgets.QComboBox(self.groupBox)
        self.num.setGeometry(QtCore.QRect(40, 20, 61, 22))
        self.num.setObjectName("num")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.num.addItem("")
        self.findExt = QtWidgets.QPushButton(self.groupBox)
        self.findExt.setGeometry(QtCore.QRect(10, 50, 121, 23))
        self.findExt.setObjectName("findExt")
        self.findInst = QtWidgets.QPushButton(self.one)
        self.findInst.setGeometry(QtCore.QRect(20, 70, 121, 23))
        self.findInst.setObjectName("findInst")
        self.findDist = QtWidgets.QPushButton(self.one)
        self.findDist.setGeometry(QtCore.QRect(20, 190, 121, 23))
        self.findDist.setObjectName("findDist")
        self.showResult = QtWidgets.QPushButton(self.one)
        self.showResult.setGeometry(QtCore.QRect(20, 230, 121, 23))
        self.showResult.setObjectName("showResult")
        self.ar = QtWidgets.QGroupBox(self.centralwidget)
        self.ar.setGeometry(QtCore.QRect(430, 40, 161, 261))
        self.ar.setObjectName("ar")
        self.showOnBoard = QtWidgets.QPushButton(self.ar)
        self.showOnBoard.setGeometry(QtCore.QRect(10, 100, 141, 23))
        self.showOnBoard.setObjectName("showOnBoard")
        self.showVertical = QtWidgets.QPushButton(self.ar)
        self.showVertical.setGeometry(QtCore.QRect(10, 160, 141, 23))
        self.showVertical.setObjectName("showVertical")
        self.text = QtWidgets.QTextEdit(self.ar)
        self.text.setGeometry(QtCore.QRect(20, 40, 121, 31))
        self.text.setObjectName("text")
        self.label_3 = QtWidgets.QLabel(self.ar)
        self.label_3.setGeometry(QtCore.QRect(60, 20, 71, 16))
        self.label_3.setObjectName("label_3")
        self.stereo = QtWidgets.QGroupBox(self.centralwidget)
        self.stereo.setGeometry(QtCore.QRect(630, 40, 161, 261))
        self.stereo.setObjectName("stereo")
        self.stereoMap = QtWidgets.QPushButton(self.stereo)
        self.stereoMap.setGeometry(QtCore.QRect(10, 120, 141, 23))
        self.stereoMap.setObjectName("stereoMap")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 181, 16))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 830, 21))
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
        self.Load.setTitle(_translate("MainWindow", "Load Image"))
        self.loadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.LoadL.setText(_translate("MainWindow", "Load Image_L"))
        self.loadR.setText(_translate("MainWindow", "Load Image_R"))
        self.one.setTitle(_translate("MainWindow", "1.Calibration"))
        self.findCorner.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.groupBox.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.num.setItemText(0, _translate("MainWindow", "1"))
        self.num.setItemText(1, _translate("MainWindow", "2"))
        self.num.setItemText(2, _translate("MainWindow", "3"))
        self.num.setItemText(3, _translate("MainWindow", "4"))
        self.num.setItemText(4, _translate("MainWindow", "5"))
        self.num.setItemText(5, _translate("MainWindow", "6"))
        self.num.setItemText(6, _translate("MainWindow", "7"))
        self.num.setItemText(7, _translate("MainWindow", "8"))
        self.num.setItemText(8, _translate("MainWindow", "9"))
        self.num.setItemText(9, _translate("MainWindow", "10"))
        self.num.setItemText(10, _translate("MainWindow", "11"))
        self.num.setItemText(11, _translate("MainWindow", "12"))
        self.num.setItemText(12, _translate("MainWindow", "13"))
        self.num.setItemText(13, _translate("MainWindow", "14"))
        self.num.setItemText(14, _translate("MainWindow", "15"))
        self.findExt.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.findInst.setText(_translate("MainWindow", "1.2 Find Instrinsic"))
        self.findDist.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.showResult.setText(_translate("MainWindow", "1.5 Show Result"))
        self.ar.setTitle(_translate("MainWindow", "2.Augmented Reality"))
        self.showOnBoard.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.showVertical.setText(_translate("MainWindow", "2.2 Show Words Vertically"))
        self.label_3.setText(_translate("MainWindow", " 6 letters"))
        self.stereo.setTitle(_translate("MainWindow", "3.Stereo Disparity Map"))
        self.stereoMap.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))
        self.label.setText(_translate("MainWindow", "P96111119 謝宗佑"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

