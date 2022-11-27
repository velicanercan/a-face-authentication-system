# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:15:18 2022

@author: Velican
"""
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication, QWidget
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from thread import VideoThread
import json


class NewUser(QMainWindow):
    def __init__(self):
        super().__init__()
        QMainWindow.__init__(self)
        loadUi(paths["UserUIPath"], self)
        self.feat = None
        self.disply_width = 400
        self.display_height = 400
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.cancelBtn.clicked.connect(self.close)
        self.startButton.clicked.connect(self.dataset)
        self.testButton.clicked.connect(self.predictor)
        self.appendButton.clicked.connect(self.appendFeatures)

    def dataset(self, host):
        face_classifier = cv2.CascadeClassifier(paths["HaarCascadePath"])

        def face_crop(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if faces == ():
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        host = self.userName.text()
        path = os.path.join(paths["UserDatasetPath"], host)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            pass
        video_capture = cv2.VideoCapture(0)
        img_id = 0
        while True:
            # capture frame-by-frame
            ret, frame = video_capture.read()
            if face_crop(frame) is not None:
                img_id += 1
                face = cv2.resize(face_crop(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = path + "/" + host + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.imshow("", face)
                cv2.moveWindow("", 695, 200)
            if cv2.waitKey(1) == 27 or img_id == 100:
                QMessageBox.information(self, "Info", "Collection is done")
                cv2.destroyAllWindows()
                video_capture.release()
                self.thread.start()

    def predictor(self, path):
        def plot_sample(image, keypoint, axis, title):
            image = image.reshape(96, 96)
            axis.imshow(image, cmap='gray')
            axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20, color="red")
            plt.title(title)

        chosen = self.modelFile.currentText()

        model = load_model("{}.hdf5".format(paths["MLModelsPath"] + chosen))
        images = []
        path = paths["UserDatasetPath"] + self.userName.text() + "/" + self.userName.text()

        for i in range(1, 101):
            images.append(np.array(Image.open(path + "." + str(i) + ".jpg").resize((96, 96))))
        img = np.reshape(images, (-1, 96, 96, 1)) / 255.
        me = model.predict(img)
        fig = plt.figure(figsize=(16, 7))
        for i in range(20):
            axis = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
            plot_sample(img[i * 5], me[i * 5], axis, "")
        plt.show()
        self.thread.start()

    def feature_extraction(self, model, dataPath):
        chosen = self.modelFile.currentText()
        model = load_model("{}.hdf5".format(paths["MLModelsPath"] + chosen))
        label = dataPath.split("/")[-1]
        images = []
        for i in range(1, 101):
            images.append(np.array(Image.open(dataPath + "." + str(i) + ".jpg").resize((96, 96))))
        img = np.reshape(images, (-1, 96, 96, 1)) / 255.
        features = pd.DataFrame(columns=(["id", "a1", "a2", "a3", "a4", "a5", "a6"]))
        for i in range(1, 91):
            keypoint_arr = model.predict(img)[i]

            left_eye_center_x = keypoint_arr[0]
            left_eye_center_y = keypoint_arr[1]
            left_eye_inner_corner_x = keypoint_arr[4]
            left_eye_inner_corner_y = keypoint_arr[5]
            left_eye_outer_corner_x = keypoint_arr[6]
            left_eye_outer_corner_y = keypoint_arr[7]
            right_eye_inner_corner_x = keypoint_arr[8]
            right_eye_inner_corner_y = keypoint_arr[9]
            right_eye_outer_corner_x = keypoint_arr[10]
            right_eye_outer_corner_y = keypoint_arr[11]
            left_eyebrow_inner_end_x = keypoint_arr[12]
            left_eyebrow_inner_end_y = keypoint_arr[13]
            left_eyebrow_outer_end_x = keypoint_arr[14]
            left_eyebrow_outer_end_y = keypoint_arr[15]
            right_eyebrow_inner_end_x = keypoint_arr[16]
            right_eyebrow_inner_end_y = keypoint_arr[17]
            right_eyebrow_outer_end_x = keypoint_arr[18]
            right_eyebrow_outer_end_y = keypoint_arr[19]
            nose_tip_x = keypoint_arr[20]
            nose_tip_y = keypoint_arr[21]
            mouth_left_corner_x = keypoint_arr[22]
            mouth_left_corner_y = keypoint_arr[23]
            mouth_right_corner_x = keypoint_arr[24]
            mouth_right_corner_y = keypoint_arr[25]

            eyebrow_mid = np.sqrt(np.square(left_eyebrow_inner_end_x - right_eyebrow_inner_end_x) + np.square(
                left_eyebrow_inner_end_y - right_eyebrow_inner_end_y))
            eyebrow_left = np.sqrt(np.square(left_eyebrow_inner_end_x - left_eyebrow_outer_end_x) + np.square(
                left_eyebrow_inner_end_y - left_eyebrow_outer_end_y))
            eyebrow_right = np.sqrt(np.square(right_eyebrow_inner_end_x - right_eyebrow_outer_end_x) + np.square(
                right_eyebrow_inner_end_y - right_eyebrow_outer_end_y))

            eyebrow_left_out_to_nose = np.sqrt(
                np.square(left_eyebrow_outer_end_x - nose_tip_x) + np.square(left_eyebrow_outer_end_y - nose_tip_y))
            eyebrow_left_in_to_nose = np.sqrt(
                np.square(left_eyebrow_inner_end_x - nose_tip_x) + np.square(left_eyebrow_inner_end_y - nose_tip_y))
            eyebrow_right_in_to_nose = np.sqrt(
                np.square(right_eyebrow_inner_end_x - nose_tip_x) + np.square(right_eyebrow_inner_end_y - nose_tip_y))

            eyebrow_right_out_to_mouth_right = np.sqrt(
                np.square(right_eyebrow_outer_end_x - mouth_right_corner_x) + np.square(
                    right_eyebrow_outer_end_y - mouth_right_corner_y))
            eyebrow_right_in_to_mouth_right = np.sqrt(
                np.square(right_eyebrow_inner_end_x - mouth_right_corner_x) + np.square(
                    right_eyebrow_inner_end_y - mouth_right_corner_y))

            eyebrow_left_out_to_mouth_left = np.sqrt(
                np.square(left_eyebrow_outer_end_x - mouth_left_corner_x) + np.square(
                    left_eyebrow_outer_end_y - mouth_left_corner_y))

            nose_to_mouth_right = np.sqrt(
                np.square(nose_tip_x - mouth_right_corner_x) + np.square(nose_tip_y - mouth_right_corner_y))
            nose_to_mouth_left = np.sqrt(
                np.square(nose_tip_x - mouth_left_corner_x) + np.square(nose_tip_y - mouth_left_corner_y))

            mouth_width = np.sqrt(np.square(mouth_left_corner_x - mouth_right_corner_x) + np.square(
                mouth_left_corner_y - mouth_right_corner_y))

            eyebrow_mid_x = (left_eyebrow_outer_end_x + right_eyebrow_inner_end_x) / 2
            eyebrow_mid_y = (left_eyebrow_outer_end_y + right_eyebrow_inner_end_y) / 2

            eyebrow_distance = np.sqrt(np.square(right_eyebrow_outer_end_x - left_eyebrow_inner_end_x) + np.square(
                right_eyebrow_outer_end_y - left_eyebrow_inner_end_y))
            eyebrowR = np.sqrt(np.square(right_eyebrow_inner_end_x - right_eyebrow_outer_end_x) + np.square(
                right_eyebrow_inner_end_y - right_eyebrow_outer_end_y))

            eyebrow_mid_to_nose = np.sqrt(np.square(nose_tip_x - eyebrow_mid_x) + np.square(nose_tip_y - eyebrow_mid_y))

            mouth_width = np.sqrt(np.square(mouth_left_corner_x - mouth_right_corner_x) + np.square(
                mouth_left_corner_y - mouth_right_corner_y))

            nose_tip_to_eyebrow_mid = np.sqrt(
                np.square(eyebrow_mid_x - nose_tip_x) + np.square(eyebrow_mid_y - nose_tip_y))

            eye_socketL_to_eye_center = np.sqrt(np.square(left_eye_outer_corner_x - left_eye_center_x) + np.square(
                left_eye_outer_corner_y - left_eye_center_y))

            eye_socketL_to_eye_center = np.sqrt(np.square(left_eye_outer_corner_x - left_eye_center_x) + np.square(
                left_eye_outer_corner_y - left_eye_center_y))

            if abs(right_eye_inner_corner_x - left_eye_inner_corner_x) >= 16 and eye_socketL_to_eye_center > 5:
                a1 = eyebrow_mid_to_nose / eyebrow_left_out_to_mouth_left
                a2 = nose_to_mouth_right / eyebrow_distance
                a3 = mouth_width / eyebrow_mid_to_nose
                a4 = eyebrow_distance / eyebrow_right_in_to_mouth_right
                a5 = nose_to_mouth_left / nose_tip_to_eyebrow_mid
                a6 = eyebrowR / mouth_width

                features = features.append({"id": label, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6},
                                           ignore_index=True)
            else:
                pass
        return features

    def appendFeatures(self):
        chosen = self.modelFile.currentText()
        model = load_model("{}.hdf5".format(paths["MLModelsPath"] + chosen))
        dataPath = self.userName.text() + "/" + self.userName.text()
        df = self.feature_extraction(model, paths["UserDatasetPath"]+dataPath)
        df.to_csv(paths["TrainingFeatures"], mode='a', index=False, header=False, float_format='%.2f')
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.label_2.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


data = open('paths.json')
paths = json.load(data)
