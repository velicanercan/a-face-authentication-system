# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:49:20 2022

@author: Velican
"""
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from webcam
        self._run_flag = True
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()  
            self.change_pixmap_signal.emit(cv_img)
            if cv2.waitKey(1)==27:
                self._run_flag=False
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
