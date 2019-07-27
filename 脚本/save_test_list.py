import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re


def search_jpg(label_dir):
    label_ext='.jpg'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label
	
label=search_jpg('/mnt/h/darknet/data/coco/val2017bak')
sorted(label)
list_file = open('test.txt', 'w')
for image_id in label:
    newid = image_id.strip('.xml')
    list_file.write('/mnt/h/darknet/data/coco/val2017bak/%s.jpg\n'%newid)
list_file.close()


