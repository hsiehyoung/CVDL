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
        MainWindow.resize(563, 389)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 521, 331))
        self.groupBox.setObjectName("groupBox")
        self.load = QtWidgets.QPushButton(self.groupBox)
        self.load.setGeometry(QtCore.QRect(20, 30, 191, 31))
        self.load.setObjectName("load")
        self.showTrain = QtWidgets.QPushButton(self.groupBox)
        self.showTrain.setGeometry(QtCore.QRect(20, 80, 191, 31))
        self.showTrain.setObjectName("showTrain")
        self.showModel = QtWidgets.QPushButton(self.groupBox)
        self.showModel.setGeometry(QtCore.QRect(20, 130, 191, 31))
        self.showModel.setObjectName("showModel")
        self.showData = QtWidgets.QPushButton(self.groupBox)
        self.showData.setGeometry(QtCore.QRect(20, 180, 191, 31))
        self.showData.setObjectName("showData")
        self.showAccuracy = QtWidgets.QPushButton(self.groupBox)
        self.showAccuracy.setGeometry(QtCore.QRect(20, 230, 191, 31))
        self.showAccuracy.setObjectName("showAccuracy")
        self.inference = QtWidgets.QPushButton(self.groupBox)
        self.inference.setGeometry(QtCore.QRect(20, 280, 191, 31))
        self.inference.setObjectName("inference")
        self.show = QtWidgets.QGraphicsView(self.groupBox)
        self.show.setGeometry(QtCore.QRect(240, 60, 256, 241))
        self.show.setObjectName("show")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(240, 10, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 563, 21))
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
        self.groupBox.setTitle(_translate("MainWindow", "5.Resnet101 Test"))
        self.load.setText(_translate("MainWindow", "Load Image"))
        self.showTrain.setText(_translate("MainWindow", "1. Show Train Images"))
        self.showModel.setText(_translate("MainWindow", "2. Show Model Structure"))
        self.showData.setText(_translate("MainWindow", "3. Show Data Augmentation"))
        self.showAccuracy.setText(_translate("MainWindow", "4. Show Accuracy and Loss"))
        self.inference.setText(_translate("MainWindow", "5. Inference"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

