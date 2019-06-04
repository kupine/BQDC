import cv2
 
img = cv2.imread("test.jpg")
print(img.shape)
cropped = img[0:128, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("cv_cut_test.jpg", cropped)
