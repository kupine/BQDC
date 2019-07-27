import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re

#classes = ["person","suitcase","train","handbag","backpack","barrel","box","Hammer","Basket","pole","Spade","umbrella","mop","CableDrum","basketball","football","tennis","volleyball"]

classes = ["box","train"]

sets=['train', 'val']
#sets=['train']


def search_jpg(label_dir):
    label_ext='.jpg'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

wd = getcwd()

for set in sets:
    for classname in classes:
        label=search_jpg('%s/JPEGImages/%s/%s'%(wd,set,classname))
        sorted(label)
        list_file = open('%s.txt'%(set), 'a')
        for image_id in label:
            newid = image_id.strip('.jpg')
            list_file.write('%s/JPEGImages/%s/%s/%s.jpg\n'%(wd,set,classname,newid))
        list_file.close()


