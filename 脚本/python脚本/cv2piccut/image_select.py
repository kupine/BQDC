import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile

#proportions = [1,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8]


def search_jpg(label_dir):
    label_ext='.jpg'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

#for proportion in proportions:
label = search_jpg('D:/MATLAB/matting/box')
sorted(label)
for img_id in label:
    newid = img_id.strip('.jpg')
    copyfile('F:/NCAC_img_boxresmall/Annotations/box/%s.xml' %newid, 'D:/MATLAB/matting/Annotations/box/%s.xml' % newid)

print(11111)