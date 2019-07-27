#coding:utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile

#sets=['train', 'val']

sets=['train']

#classes = ["n04540053"]
#classes = ["n02747177","n02769748","n02795169","n02802426","n02909870","n02917067","n02971356","n03014705","n03127925","n03481172","n03482405","n03764736","n03976657","n04049303","n04204238","n04208210","n04254680","n04310018","n04311174","n04335435","n04367480","n04409515","n04487081","n04507155","n04540053","CableDrum"]

#classes = ["n02802426","n04254680","n04540053","n02971356"]

classes = ["suitcase"]

#filenames =['barrel','backpack','barrel','basketball','barrel','train','box','box','box','Hammer','Basket','barrel','pole','barrel','Basket','Spade','football','train','barrel','train','mop','tennis','train','umbrella','volleyball','CableDrum']

#filenames = ['basketball','football','tennis','volleyball',"CableDrum"]
#filenames = ['ball','ball','ball',"box"]

filenames = ["suitcase"]

#clsnum = [6,5,6,15,6,2,7,7,7,8,9,6,10,6,9,11,0,2,6,2,13,2,2,12,4,14]

#clsnum = [15,16,17,18,14]

#clsnum = [0,0,0,1]

clsnum = [1]

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

def convert_annotation(image_set,image_id,filename):

    in_file = open('Annotations/%s/%s/%s.xml'%(image_set,filename, image_id), encoding='UTF-8')
    print (image_id)
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
    
    for obj in root.iter('object'):
        for classname in classes:
            listid = classes.index(classname)
            cls_id = clsnum[listid]
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls == classname:
                out_file = open('labels/%s/%s/%s.txt'%(image_set,filename,image_id), 'a')
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb]) + "\n")
        #i = i + 1
    #if i > 0:
                copyfile('JPEGImages/%s/%s/%s.jpg'%(image_set,filename,image_id), 'nJPEGImages/%s/%s/%s.jpg'%(image_set,filename,image_id))
                copyfile('Annotations/%s/%s/%s.xml'%(image_set,classname,image_id), 'nAnnotations/%s/%s/%s.xml'%(image_set,filename,image_id))
wd = getcwd()

def search_jpg(label_dir):
    label_ext='.jpg'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    #print (label)
    return label


for image_set in sets:
    #for classname in classes:
    #    listid = classes.index(classname)
    filename = filenames[0]
    #    cls_id = clsnum[listid]
    label=search_jpg('JPEGImages/%s/suitcase'%(image_set))
    sorted(label)
    for image_id in label:
        newid = image_id.strip('.jpg')
        convert_annotation(image_set,newid,filename)

