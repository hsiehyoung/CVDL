# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:50:19 2022

@author: 謝宗佑
"""

from PyQt5 import QtWidgets

from controller import MainWindow_controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())