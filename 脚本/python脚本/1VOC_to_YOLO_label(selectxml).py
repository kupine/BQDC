#encoding: 'UTF-8'

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile

sets=['train2017','val2017']

#classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","fire hydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","couch","pottedplant","bed","diningtable","toilet","tv","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
#classes = ["person","suitcase","train","stone","ball","CableDrum","handbag","sportsball","backpack","hairdrier"]
classes = ["person","suitcase","train"]
#classes = ["person","suitcase","train","handbag","sportsball","backpack","barrel","box","Hammer","Basket","pole","Spade","umbrella","mop","CableDrum"]

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

def convert_annotation(image_set,image_id,classname):

    in_file = open('Annotations/%s/%s.xml'%(image_set, image_id),'rb')
    #out_file = open('coco/%s/%s/%s.txt'%(image_set,classname,image_id), 'w')
    txtpath='%s/suitcase/labels/%s/%s/'%(wd,image_set,classname)
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
    xmlpath='%s/suitcase/Annotations/%s/%s/'%(wd,image_set,classname)
    if not os.path.exists(xmlpath):
        os.makedirs(xmlpath)
    jpgpath='%s/suitcase/JPEGImages/%s/%s/'%(wd,image_set,classname)
    if not os.path.exists(jpgpath):
        os.makedirs(jpgpath)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
       # if cls not in classes or int(difficult)==1:
        if cls == classname:
            copyfile('Annotations/%s/%s.xml'%(image_set, image_id), 'suitcase/Annotations/%s/%s/%s.xml'%(image_set,classname,image_id))
            #in_xmlfile = open('coco/Annotations/%s/%s/%s%s.xml'%(image_set,classname,classname,image_id))
            tree=ET.parse('suitcase/Annotations/%s/%s/%s.xml'%(image_set,classname,image_id))
            root = tree.getroot()
            for object in root.findall('object'):
                cls = object.find('name').text
                if cls != classname:
                    root.remove(object)
                #print (cls)
            tree.write('suitcase/Annotations/%s/%s/%s.xml'%(image_set,classname,image_id))
            #in_xmlfile.close()
            #out_file = open('coco/labels/%s/%s/%s_%s.txt'%(image_set,classname,classname,image_id), 'a')
         #continue
            #cls_id = classes.index(cls)
            #xmlbox = obj.find('bndbox')
            #b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            #bb = convert((w,h), b)
            #out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb]) + "\n")
            copyfile('%s/%s/%s.jpg'%(image_set,image_set,image_id), 'suitcase/JPEGImages/%s/%s/%s.jpg'%(image_set,classname,image_id))

wd = getcwd()

def search_xml(label_dir):
    label_ext='.xml'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

for image_set in sets:
    for classname in classes:
    #if not os.path.exists('Annotations/labels/'):
        #os.makedirs('Annotations/labels/')
    #image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        label=search_xml('Annotations/%s'%image_set)
        sorted(label)
        list_file = open('%s.txt'%(image_set), 'w')
        for image_id in label:
            newid = image_id.strip('.xml')
            #print (newid)
            list_file.write('%s/JPEGImages/%s/%s/%s.jpg\n'%(wd,image_set,classname,newid))
            convert_annotation(image_set,newid,classname)
        list_file.close()

#os.system("cat train2017.txt val2017.txt > train.txt")
#os.system("cat 2007_val.txt 2012_val.txt > val.txt")

