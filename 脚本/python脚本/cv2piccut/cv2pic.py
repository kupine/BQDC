import cv2
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile

wd = getcwd()

jpgpath='%s/BJtestPIC/'%(wd)
if not os.path.exists(jpgpath):
    os.makedirs(jpgpath)
njpgpath='%s/nBJtestPIC/'%(wd)
if not os.path.exists(njpgpath):
    os.makedirs(njpgpath)

def search_jpg(label_dir):
    label_ext='.jpg'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

label=search_jpg("%s/BJtestPIC/"%(wd))
sorted(label)
for image_id in label:
    #newid = image_id.strip('.jpg')
    print (image_id)

    img = cv2.imread("%s/BJtestPIC/%s"%(wd,image_id),2 | 4)
    print(img.shape)
    cropped = img[0:1080, 420:1500]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("%s/nBJtestPIC/%s"%(wd,image_id), cropped,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
