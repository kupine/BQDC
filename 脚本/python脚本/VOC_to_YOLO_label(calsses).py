import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re

sets=['train2017', 'val2017']

#classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
#classes = ["person","suitcase","train","stone","ball","CableDrum","handbag","sportsball","backpack"]
classes = ["sportsball"]

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

    in_file = open('Annotations/%s/%s.xml'%(image_set, image_id))
    #out_file = open('coco/%s/%s/%s.txt'%(image_set,classname,image_id), 'w')
    txtpath='%s/coco/labels/%s/%s/'%(wd,image_set,classname)
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
    jpgpath='%s/coco/JPEGImages/%s/%s/'%(wd,image_set,classname)
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
            out_file = open('coco/labels/%s/%s/%s.txt'%(image_set,classname,image_id), 'a')
         #   continue
            cls_id = classes.index(cls)
            #cls_id = 7
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb]) + "\n")

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
            print (newid)
            list_file.write('%s/JPEGImages/%s/%s/%s.jpg\n'%(wd,image_set,classname,newid))
            convert_annotation(image_set,newid,classname)
        list_file.close()

#os.system("cat train2017.txt val2017.txt > train.txt")
#os.system("cat 2007_val.txt 2012_val.txt > val.txt")

