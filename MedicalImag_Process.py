
# -*- coding: utf-8 -*-
"""

@author: kenni.konate
"""

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, QFileDialog, QAction, QMessageBox, QWidget, QVBoxLayout, QPushButton,
QInputDialog, QLineEdit, QGridLayout)
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PySide2.QtCore import Slot
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
import os


#load ui file
uifile_1 = 'program_interface.ui'
form_1, base_1 = uic.loadUiType(uifile_1)



class MyWindow(base_1, form_1):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setupUi(self)
        uic.loadUi('program_interface.ui',self)
        self.actionAc.triggered.connect(self.openImage)
        self.actionKaydet.triggered.connect(self.save)
        self.actionCikis.triggered.connect(self.close)
        #Filter
        self.actionOrtalama_Mean.triggered.connect(self.MeanFilter)
        self.actionOrtanca_Filtre_Median.triggered.connect(self.MedianFilter)
        self.actionAsinma.triggered.connect(self.erosion)
        self.actionGenisleme.triggered.connect(self.dilation)
        self.actionAcinim.triggered.connect(self.opening)
        self.actionKapanim.triggered.connect(self.closing)
        self.actionTek_nokta_bul.triggered.connect(self.Hit_Miss_Single_point)
        self.actionUc_nokta_bul.triggered.connect(self.Hit_Miss_endpoints)
        self.actionKesisim_nokta_bul.triggered.connect(self.Hit_Miss_FindIntersectionPoint)
#        self.actiongenisletme.triggered.connect(self.growing_regionMain)
        
        
    def openImage(self):
        global imagePath
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        global img
        img = cv2.imread(imagePath)
        img=cv2.resize(img,(256,256))
        cv2.imshow('Original image',img)
        if len(img.shape) ==3:
            self.actionGri_Seviye_Donusum.triggered.connect(self.grayImageConvert)
    
    def save(self):
        cv2.imwrite(img,img)
        
    def saveAs(self):
        textboxValue = self.textbox.text()
        cv2.imwrite(textboxValue,img)
    
    def close(self):
        sys.exit()
        
    
    def grayImageConvert(self):
        global gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#        print('channels', gray.shape)
        cv2.imshow('Gray image', gray)
        self.actionHistogram_Grafigi.triggered.connect(self.HistogramGraf)
        self.actionEsikleme.triggered.connect(self.esikleme)
        self.actionHistogram_E_itleme.triggered.connect(self.HitogramEqualization)
        
    
    def HistogramGraf(self):
        hist,bins = np.histogram(gray.flatten(),256,[0,256])
        pg.plot(hist)
    
    def esikleme(self):
        retval, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Threshold', threshold)
    
    def HitogramEqualization(self):
        equalize_image = cv2.equalizeHist(gray)
        cv2.imshow('Image Equalized', equalize_image)
        hist,bins = np.histogram(equalize_image.flatten(),256,[0,256])
        pg.plot(hist)
    
    #    def ContrastExtension(self):
    
    def MeanFilter(self):
        image_mean_filter=cv2.blur(gray,(3,3))
        cv2.imshow('mean Filter', image_mean_filter)
    
    def MedianFilter(self):
        image_median_filter=cv2.medianBlur(gray, 3)
        cv2.imshow('Median Filter', image_median_filter)
    
        
    def ConvertImgBlackWhite(self):
        imge = cv2.imread(imagePath,0)
        resized = cv2.resize(imge,(256,256))
        global BlackWhite_img
        if len(imge.shape)== 2:
            thresh, BlackWhite_img = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow('BlackWhite_img',BlackWhite_img)
    
    def erosion(self):
         self.ConvertImgBlackWhite()
         kernel = np.ones((3,3),np.uint8)
         erosion = cv2.erode(BlackWhite_img,kernel,iterations = 1)
         cv2.imshow('Erosion',erosion)
            
    def dilation(self):
        self.ConvertImgBlackWhite()
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(BlackWhite_img,kernel,iterations = 1)
        cv2.imshow('Dilation',dilation)
    def opening(self):
        self.ConvertImgBlackWhite()
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(BlackWhite_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Opening',opening)
    def closing(self):
        self.ConvertImgBlackWhite()
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(BlackWhite_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Closing',closing)
    
    def Hit_Miss_Single_point(self):
        self.ConvertImgBlackWhite()
        kernel = np.array((
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]), dtype="int")
        
        output_image = cv2.morphologyEx(BlackWhite_img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('HitAndMissSinglePoint', output_image)
    
    def Hit_Miss_endpoints(self):
        self.ConvertImgBlackWhite()
        kernel = np.array((
        [1 , 1, 1],
        [1, 0, 1],
        [1, 1, 1]), dtype="int")
        
        output_image = cv2.morphologyEx(BlackWhite_img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('HitAndMissEndPoints', output_image)
    
    def Hit_Miss_FindIntersectionPoint(self):
        self.ConvertImgBlackWhite()
        kernel = np.array((
        [0 , 1, 1],
        [1, 0, 1],
        [0, 1, 0]), dtype="int")
        
        output_image = cv2.morphologyEx(BlackWhite_img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('HitAndMissIntersectionPoints', output_image)    
            
    def watershed(self):
        self.ConvertImgBlackWhite.actionHistogram_E_itleme.triggered.connect(self.HitogramEqualization)
        self.actionAcinim.triggered.connect(self.opening)
        opening=self.opening
        ret, markers = cv2.connectedComponents(opening)
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]
        cv2.imshow('Watershed', img)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
