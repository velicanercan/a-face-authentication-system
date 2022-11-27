# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/Velican/Desktop/UI/homeScreen.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 480)
        MainWindow.setMinimumSize(QtCore.QSize(800, 480))
        MainWindow.setMaximumSize(QtCore.QSize(800, 480))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 20, 671, 251))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("1d5ebd5a-7bef-4aa4-ab48-c893eebc041e.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 320, 701, 101))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(4, 4, 4, 4)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.newUserButton = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.newUserButton.sizePolicy().hasHeightForWidth())
        self.newUserButton.setSizePolicy(sizePolicy)
        self.newUserButton.setStyleSheet("#newUserButton{font: 10pt \"MS Shell Dlg 2\" white ;\n"
"border-radius: 10px ;\n"
"background-color: rgb(172, 24, 45);\n"
"color: rgb(255, 255, 255);\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"\n"
"#newUserButton:hover{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0.453136, x2:1, y2:0.523273, stop:0 rgba(104, 113, 138, 255), stop:1 rgba(215, 221, 232, 255));\n"
"}")
        self.newUserButton.setObjectName("newUserButton")
        self.horizontalLayout.addWidget(self.newUserButton)
        self.verificationButton = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.verificationButton.sizePolicy().hasHeightForWidth())
        self.verificationButton.setSizePolicy(sizePolicy)
        self.verificationButton.setStyleSheet("#verificationButton{font: 10pt \"MS Shell Dlg 2\" white ;\n"
"border-radius: 10px ;\n"
"background-color: rgb(172, 24, 45);\n"
"color: rgb(255, 255, 255);\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"\n"
"#verificationButton:hover{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0.453136, x2:1, y2:0.523273, stop:0 rgba(104, 113, 138, 255), stop:1 rgba(215, 221, 232, 255));\n"
"}")
        self.verificationButton.setObjectName("verificationButton")
        self.horizontalLayout.addWidget(self.verificationButton)
        self.doorBellButton = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doorBellButton.sizePolicy().hasHeightForWidth())
        self.doorBellButton.setSizePolicy(sizePolicy)
        self.doorBellButton.setStyleSheet("#doorBellButton{font: 10pt \"MS Shell Dlg 2\" white ;\n"
"border-radius: 10px ;\n"
"background-color: rgb(172, 24, 45);\n"
"color: rgb(255, 255, 255);\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"\n"
"#doorBellButton:hover{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0.453136, x2:1, y2:0.523273, stop:0 rgba(104, 113, 138, 255), stop:1 rgba(215, 221, 232, 255));\n"
"}")
        self.doorBellButton.setObjectName("doorBellButton")
        self.horizontalLayout.addWidget(self.doorBellButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
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
        self.newUserButton.setText(_translate("MainWindow", "New User"))
        self.verificationButton.setText(_translate("MainWindow", "Verification "))
        self.doorBellButton.setText(_translate("MainWindow", "Door-Bell"))

