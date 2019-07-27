#coding:utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile

sets=['train', 'val']

#classes = ["n04540053"]
classes = ["n02917067","n02971356","n03127925","n04310018"]

filenames =['train','box','box','train']

clsnum = [2,3,3,2]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_set,image_id,classname,cls_id,filename):

    in_file = open('Annotations/%s/%s/%s.xml'%(image_set,classname, image_id))
    print (image_id)
    #out_file = open('ncac/%s/%s/%s.txt'%(image_set,classname,image_id), 'w')
    txtpath='%s/labels/%s/%s/'%(wd,image_set,filename)
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
    jpgpath='%s/nJPEGImages/%s/%s/'%(wd,image_set,filename)
    if not os.path.exists(jpgpath):
        os.makedirs(jpgpath)
    xmlpath='%s/nAnnotations/%s/%s/'%(wd,image_set,filename)
    if not os.path.exists(xmlpath):
        os.makedirs(xmlpath)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    i = 0
    #list_file = open('ncac%s.txt'%(image_set), 'a')
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
       # if cls not in classes or int(difficult)==1:
        if cls == classname:
            out_file = open('labels/%s/%s/%s.txt'%(image_set,filename,image_id), 'a')
         #   continue
            #cls_id = classes.index(cls)
            #cls_id = clsnum
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb]) + "\n")
            i = i + 1
    if i > 0:
        copyfile('JPEGImages/%s/%s/%s.jpg'%(image_set,classname,image_id), 'nJPEGImages/%s/%s/%s.jpg'%(image_set,filename,image_id))
        copyfile('Annotations/%s/%s/%s.xml'%(image_set,classname,image_id), 'nAnnotations/%s/%s/%s.xml'%(image_set,filename,image_id))
        #list_file.write('%s/JPEGImages/%s/%s/%s.jpg\n'%(wd,image_set,classname,image_id))
        #print (classname)
        #list_file.close()
wd = getcwd()

def search_xml(label_dir):
    label_ext='.xml'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

for image_set in sets:
    for classname in classes:
        listid = classes.index(classname)
        filename = filenames[listid]
        cls_id = clsnum[listid]
    #if not os.path.exists('Annotations/labels/'):
        #os.makedirs('Annotations/labels/')
    #image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        label=search_xml('Annotations/%s/%s'%(image_set,classname))
        sorted(label)
        #list_file = open('ncac%s.txt'%(image_set), 'w')
        for image_id in label:
            newid = image_id.strip('.xml')
            #print (newid)
            #list_file.write('%s/ncac/JPEGImages/%s/%s/%s%s.jpg\n'%(wd,image_set,classname,classname,newid))
            convert_annotation(image_set,newid,classname,cls_id,filename)
        #list_file.close()

#os.system("cat train2017.txt val2017.txt > train.txt")
#os.system("cat 2007_val.txt 2012_val.txt > val.txt")

