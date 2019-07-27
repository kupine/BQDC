#筛选指定分类的xml文件，并缩放图片以及标签的大小
import xml.etree.ElementTree as ET
import pickle
import os,shutil
from os import listdir, getcwd
from os.path import join
import sys
import lxml

#year_sets=["2012","2007"]
image_sets=["train"]
#class_sets = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#class_sets = ["person", "train"]
class_sets = ["suitcase"]
proportion = 0.1#缩放的系数

wd = os.getcwd()

nxmlpath='F:/NCAC_BJsuitcase/nAnnotations/%s/'%(image_set,classesname)
if not os.path.exists(njpgpath):
    os.makedirs(njpgpath)

annotation_path = 'F:/NCAC_BJsuitcase/Annotations/%s/'%(image_set,classesname)

def search_xml(label_dir):#遍历xml
    label_ext='.xml'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

def write_xml(annotation_path):#修改数据并保存
    label=search_xml(annotation_path)
    sorted(label)
    for index,file in  enumerate(label):
        cur_xml = ET.parse(annotation_path+file)
        root = cur_xml.getroot()
        for node in root:
            children = node.getchildren()
            if node.tag=='size':
                w=node.find('width')
                width=int(w.text)
                w.text=str(int(width*proportion))
                h=node.find('height')
                height=int(h.text)
                h.text=str(int(height*proportion))
            elif node.tag=='object':
                for child in children:
                    if child.tag=='bndbox':
                        xi=child.find('xmin')
                        xa=child.find('xmax')
                        yi=child.find('ymin')
                        ya=child.find('ymax')
                        xmin=int(xi.text)#bbox position
                        ymin=int(yi.text)
                        xmax=int(xa.text)
                        ymax=int(ya.text)
                        xi.text=str(int(xmin*proportion))
                        yi.text=str(int(ymin*proportion))
                        xa.text=str(int(xmax*proportion))
                        ya.text=str(int(ymax*proportion))
        cur_xml.write(nxmlpath+file)


for image_set in image_sets:
    for classesname in class_sets:
        write_xml(annotation_path)
