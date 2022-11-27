# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:13:36 2022

@author: Velican
"""
from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import QMainWindow,QMessageBox,QApplication,QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
import numpy as np
import cv2
import os
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier 
from datetime import datetime
import json

class Training(QMainWindow):
    def __init__(self):
        super().__init__()
        QMainWindow.__init__(self)
        loadUi(paths["TrainingUIPath"],self)      
     
        self.trainButton.clicked.connect(self.trainingBtn)
        self.loadCsvButton.clicked.connect(self.openFile)
        self.saveButton.clicked.connect(self.saveModel)
    
    def trainingBtn(self):
        df = self.all_data
        X = df.drop(["id"],axis=1)
        y = df["id"]
        
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(X)
        scaled_x = scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = self.testSize.value(),random_state = 5,shuffle=True)
        
        if self.linearSvm.isChecked():
            model_linear = SVC(kernel='linear',C=self.cValue.value(),gamma="auto",probability=True)
            model_linear.fit(X_train, y_train)
            y_pred = model_linear.predict(X_test)
            self.accuracy.setText(str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred)))
            self.classificationReport.setText(str(classification_report(y_test, y_pred)))
            self.infoBox.setText(str(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)))
        
        elif self.rbf.isChecked():
            model_rbf = SVC(kernel='rbf',C=self.cValue.value(),gamma="auto",probability=True)
            model_rbf.fit(X_train, y_train)
            y_pred = model_rbf.predict(X_test)
            self.accuracy.setText(str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred)))
            self.classificationReport.setText(str(classification_report(y_test, y_pred)))
            self.infoBox.setText(str(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)))
        
        elif self.knn.isChecked():
            model_knn = KNeighborsClassifier(n_neighbors=self.nValue.value(),metric='euclidean')
            model_knn.fit(X_train,y_train)
            y_pred = model_knn.predict(X_test)
            self.accuracy.setText(str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred)))
            self.classificationReport.setText(str(classification_report(y_test, y_pred)))
            self.infoBox.setText(str(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)))
        
        elif self.decisionTree.isChecked():   
            model_tree=DecisionTreeClassifier(criterion = 'entropy', random_state=0)
            model_tree.fit(X_train,y_train)
            y_pred = model_tree.predict(X_test)
            self.accuracy.setText(str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred)))
            self.classificationReport.setText(str(classification_report(y_test, y_pred)))
            self.infoBox.setText(str(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)))
    
    def saveModel(self):
        df = self.all_data
        X = df.drop(["id"],axis=1)
        y = df["id"]
        
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = self.testSize.value(),random_state = 5)
        
        if self.linearSvm.isChecked(): 
            model_linear = SVC(kernel='linear',C=self.cValue.value(),gamma="auto",probability=True)
            model_linear.fit(X_train, y_train)
            with open(paths["LinearSVMModel"], 'wb') as file:
                pickle.dump(model_linear, file)
        
        elif self.rbf.isChecked():
            model_rbf = SVC(kernel='rbf',C=self.cValue.value(),gamma="auto",probability=True)
            model_rbf.fit(X_train, y_train)
            with open(paths["RBFModel"], 'wb') as file:
                pickle.dump(model_rbf, file)
        
        elif self.knn.isChecked():
            model_knn = KNeighborsClassifier(n_neighbors=self.nValue.value(),metric='euclidean',algorithm="auto")
            model_knn.fit(X_train,y_train)
            with open(paths["KNNModel"], 'wb') as file:
                pickle.dump(model_knn, file)
        
        elif self.decisionTree.isChecked():   
            model_tree=DecisionTreeClassifier(criterion = 'entropy', random_state=10)
            model_tree.fit(X_train,y_train)
            with open(paths["DecisionTreeModel"], 'wb') as file:
                pickle.dump(model_tree, file)
    
    def openFile(self):
        path = QFileDialog.getOpenFileName(self,"Open CSV",os.getenv("HOME"),"CSV(*.csv)")
        self.all_data = pd.read_csv(path[0])
        self.tableWidget.setColumnCount(len(self.all_data.columns))
        self.tableWidget.setRowCount(len(self.all_data.index))
        self.tableWidget.setHorizontalHeaderLabels(self.all_data.columns)
        
        for i in range(len(self.all_data.index)):
            for j in range(len(self.all_data.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_data.iat[i, j])))

jsonData = open('paths.json')
paths = json.load(jsonData)