import numpy as np
import cv2

# 读取原始图像
img = cv2.imread('./sample.jpg')

# 显示图像
cv2.imshow("original", img)

# 定义roi列表,按(xmin,xmax,ymin,ymax)格式存放所有roi
roi_list = list()

# rois的ndarray对象
# 四个顶点坐标:依次为左上,右上,右下,左下
# 取roi区域时,只需要知道xmin,xmax,ymin,ymax即可,对应左上和右下的两个点
rois = np.array([
    [[14, 29], [499, 29], [499, 44], [14, 44]],
    [[66, 63], [275, 63], [275, 105], [66, 105]]
    ])

# 遍历rois的ndarray对象,按照指定格式存入roi_list
for roi in rois:
    roi_list.append((roi[0][0], roi[2][0], roi[0][1], roi[2][1]))

# 构建一个新的结果图像
result = np.zeros_like(img)

# 取原始图像中取对应roi数据,赋值给结果图像对应位置,注意y在前x在后
for roi in roi_list:
    result[roi[2]:roi[3], roi[0]:roi[1]] = img[roi[2]:roi[3], roi[0]:roi[1]]

# 显示结果图像
cv2.imshow("result", result)
cv2.waitKey(0)
