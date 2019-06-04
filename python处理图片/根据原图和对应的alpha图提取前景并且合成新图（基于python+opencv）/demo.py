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

#-----------------2.与新背景图像合成-----------
bg= np.zeros((3,height,width),dtype=img.dtype)  #生成背景图像
bg[2][0:height,0:width] = 255 #背景图像采用红色

dstt = np.zeros((3,height,width),dtype=img.dtype)

for i in range(3):
    dstt[i][:,:] = bg[i][:,:]*(255.0-mask)/255
    dstt[i][:,:] += np.array(img[:,:,i]*(mask/255), dtype=np.uint8)
cv2.imwrite("merge.png", cv2.merge(dstt))
