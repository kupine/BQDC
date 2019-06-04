import cv2
import numpy as np

img = cv2.imread("图二.png")#图二代表嵌入的小图片
co = cv2.imread("图一.png", -1)#图一代表一张大的背景图
scr_channels = cv2.split(co)
dstt_channels = cv2.split(img)
b, g, r, a = cv2.split(co)
for i in range(3):
    dstt_channels[i][100:200,300:400] = dstt_channels[i][100:200,300:400]*(255.0-a)/255
    dstt_channels[i][100:200,300:400] += np.array(scr_channels[i]*(a/255), dtype=np.uint8)
cv2.imwrite("img_target.png", cv2.merge(dstt_channels))
