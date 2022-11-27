from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication, QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
from new_user import *
from training import Training
from authentication import Authentication
import json


def pathExtractor():
    return json.load(open('paths.json'))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.new = None
        self.train = None
        self.auth = None
        paths = pathExtractor()
        QMainWindow.__init__(self)
        loadUi(paths["HomeUIPath"], self)

        self.newUserButton.clicked.connect(self.user)
        self.trainingButton.clicked.connect(self.training)
        self.authenticationButton.clicked.connect(self.authentication)
        
    def user(self):
        if self.new is None:
            self.new = NewUser()
        self.new.show()
    
    def training(self):
        if self.train is None:
            self.train = Training()
        self.train.show()
    
    def authentication(self):
        if self.auth is None:
            self.auth = Authentication()
        self.auth.show()


uyg = QApplication([])
window = MainWindow()
window.show()
