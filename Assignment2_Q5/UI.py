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
        MainWindow.resize(720, 472)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 231, 391))
        self.groupBox.setObjectName("groupBox")
        self.loadImage = QtWidgets.QPushButton(self.groupBox)
        self.loadImage.setGeometry(QtCore.QRect(40, 30, 151, 31))
        self.loadImage.setObjectName("loadImage")
        self.showImage = QtWidgets.QPushButton(self.groupBox)
        self.showImage.setGeometry(QtCore.QRect(40, 90, 151, 31))
        self.showImage.setObjectName("showImage")
        self.showDist = QtWidgets.QPushButton(self.groupBox)
        self.showDist.setGeometry(QtCore.QRect(40, 150, 151, 31))
        self.showDist.setObjectName("showDist")
        self.showModel = QtWidgets.QPushButton(self.groupBox)
        self.showModel.setGeometry(QtCore.QRect(40, 210, 151, 31))
        self.showModel.setObjectName("showModel")
        self.showComp = QtWidgets.QPushButton(self.groupBox)
        self.showComp.setGeometry(QtCore.QRect(40, 270, 151, 31))
        self.showComp.setObjectName("showComp")
        self.inference = QtWidgets.QPushButton(self.groupBox)
        self.inference.setGeometry(QtCore.QRect(40, 330, 151, 31))
        self.inference.setObjectName("inference")
        self.resultLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(290, 360, 391, 31))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        self.resultLabel.setFont(font)
        self.resultLabel.setText("")
        self.resultLabel.setObjectName("resultLabel")
        self.showLabel = QtWidgets.QLabel(self.centralwidget)
        self.showLabel.setGeometry(QtCore.QRect(340, 50, 291, 281))
        self.showLabel.setObjectName("showLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 720, 21))
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
        self.groupBox.setTitle(_translate("MainWindow", "ResNet50"))
        self.loadImage.setText(_translate("MainWindow", "Load Image"))
        self.showImage.setText(_translate("MainWindow", "1. Show Images"))
        self.showDist.setText(_translate("MainWindow", "2. Show Distribution"))
        self.showModel.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.showComp.setText(_translate("MainWindow", "4. Show Comparison"))
        self.inference.setText(_translate("MainWindow", "5. Inference"))
        self.showLabel.setText(_translate("MainWindow", " "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

