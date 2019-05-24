# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
import os

if __name__ == "__main__":
    imgPath='D:/Data/JPEGImages/'
    labelPath='D:/Data/Annotations/'
    flip_imgPath='D:/Data/JPEGImages_flip/'
    flip_labelPath='D:/Data/Annotations_flip/'
    for i in os.listdir(imgPath):
        name,ext=os.path.splitext(i)
        #读取图像并翻转
        srcImg = cv2.imread(imgPath + i)
        try:
            shape = srcImg.shape
        except:
            print('can not open : ' + i)
        
        width = shape[1]
        height = shape[0]
       
        flipImg = cv2.flip(srcImg,1)
        cv2.imwrite(flip_imgPath+'flip_'+i,flipImg)
        #读取xml文件并修改其中标注信息
        tree = ET.parse(labelPath + name + '.xml')
        root = tree.getroot()
        for box in root.iter('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            xmin_flip = width - xmax
            xmax_flip = width - xmin
            box.find('xmin').text = str(xmin_flip)
            box.find('xmax').text = str(xmax_flip)          
        tree.write(flip_labelPath + 'flip_' + name + '.xml')
        print(flip_labelPath + 'flip_' + name + '.xml')