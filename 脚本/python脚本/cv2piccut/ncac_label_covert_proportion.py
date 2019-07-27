#coding=utf-8
from PIL import Image#需要pillow库
import glob
from PIL import ImageFile
#筛选指定分类的xml文件，并缩放图片以及标签的大小
import xml.etree.ElementTree as ET
import pickle
import os,shutil
from os import listdir, getcwd
from os.path import join
import sys

#year_sets=["2012","2007"]
#image_sets=["train"]
#class_sets = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#class_sets = ["person", "train"]
#class_sets = ["suitcase"]
#proportion = 0.15#缩放的系数
proportions = [1,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8]
#aaa = int(proportion * 100)

wd = os.getcwd()

nxmlpath='F:/NCAC_img_boxresmall/Annotations/box/'
if not os.path.exists(nxmlpath):
    os.makedirs(nxmlpath)

def search_xml(label_dir):#遍历xml
    label_ext='.xml'
    label=[fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return label

def write_xml(annotation_path,proportion):#修改数据并保存
    label=search_xml(annotation_path)
    sorted(label)
    for index,file in  enumerate(label):
        print(file)
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

njpgpath = 'F:/NCAC_img_boxresmall/JPEGImages/box/'
if not os.path.exists(njpgpath):
    os.makedirs(njpgpath)

ImageFile.LOAD_TRUNCATED_IMAGES = True
# in_dir = os.getcwd()
# in_dir = 'F:/NCAC_img_boxresmall/%s/JPEGImages/box/'%proportion  # 当前目录
# out_dir = in_dir+' mini'
out_dir = njpgpath  # 转换后图片目录
##percent = 0.4#缩放比例
# percent = input('请输入缩放比例：')
#proportion = float(proportion)
#if not os.path.exists(out_dir): os.mkdir(out_dir)


# 图片批处理
def main(proportion):
    for files in glob.glob(in_dir + '/*.jpg'):
        filepath, filename = os.path.split(files)
        im = Image.open(files)
        w, h = im.size
        im = im.resize((int(w * proportion), int(h * proportion)))
        im.save(os.path.join(out_dir, filename))

for proportion in proportions:
    annotation_path = 'F:/NCAC_img_boxresmall/%s/Annotations/box/'%proportion
    in_dir = 'F:/NCAC_img_boxresmall/%s/JPEGImages/box/' % proportion
    proportion = float(proportion)
    main(proportion)
    write_xml(annotation_path,proportion)

print('you succeed.')
