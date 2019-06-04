# encoding:utf-8
import cv2
import numpy as np

img = cv2.imread("1.jpg")
mask = cv2.imread("2.jpg", 0) #读取灰度图像

height,width,channel=img.shape

b, g, r = cv2.split(img)

#-----------------1.获取透明的前景图像-----------
dstt = np.zeros((4,height,width),dtype=img.dtype)

dstt[0][0:height,0:width] = b
dstt[1][0:height,0:width] = g
dstt[2][0:height,0:width] = r
dstt[3][0:height,0:width] = mask
cv2.imwrite("fore.png", cv2.merge(dstt))