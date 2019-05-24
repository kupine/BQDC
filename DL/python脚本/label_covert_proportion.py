#筛选指定分类的xml文件，并缩放图片以及标签的大小
import xml.etree.ElementTree as ET
import pickle
import os,shutil
from os import listdir, getcwd
from os.path import join
import sys
import lxml

year_sets=["2012","2007"]
image_sets=["train","val"]
#class_sets = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_sets = ["person", "train"]
proportion = 0.3#缩放的系数

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
        cur_xml.write(annotation_path+file)

def convert_xml(year,xmlname):
        saveDir_ann = "./VOC%s/Annotations_Copy/"%(year)
        if not os.path.exists(saveDir_ann):
            os.mkdir(saveDir_ann) 
        fp = open('VOC%s/Annotations_Copy/%s'%(year, xmlname))        
       # classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',\
        #           'dog','horse','motorbike','pottedplant','sheep','sofa','train','tvmonitor','person']
        classes = ['person','train'] 
        lines = fp.readlines() 
        ind_start = []
        ind_end = []
        lines_id_start = lines[:]
        lines_id_end = lines[:]

        fp.close()

        fp = open('VOC%s/Annotations_Copy/%s'%(year, xmlname),'w') 
        while "\t<object>\n" in lines_id_start:
            a = lines_id_start.index("\t<object>\n")
            ind_start.append(a)
            lines_id_start[a] = "delete"
        while "\t</object>\n" in lines_id_end:
            b = lines_id_end.index("\t</object>\n")
            ind_end.append(b)
            lines_id_end[b] = "delete"     

        i=0
        mid = []
        for i in range (0 ,len(ind_start)):
            m = ind_start[i]
            for j in range(0,len(classes)):
                if  classes[j] in lines[ind_start[i]+1]:
                    while m!= ind_end[i]+1 :
                        mid.append(lines[m])
                        m +=1

        string_start = lines[0:ind_start[0]]

        if ind_end[i]+1 >= len(lines)-1:
            string_end = [lines[len(lines)-1]]
        else:
            string_end = lines[ind_end[i]+1:len(lines)]

        string_start +=mid

        string_start += string_end 
        for c in range(0,len(string_start)):
            fp.write(string_start[c])
        fp.close()

        #write_xml(saveDir_ann)

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile,dstfile)
        print ("copy %s -> %s for %s_%s in %s"%(srcfile,dstfile,classesname,image_set,year))


def select_annotation(year, image_set,classesname):
    filename = 'VOC%s/ImageSets/Main/%s_%s.txt'%(year, classesname,image_set)
    with open(filename) as file_object:
        lines = file_object.readlines()
        for line in lines:
            if line[7:9] == ' 1' and year == "2007":
                xmlname = line[0:6]+'.xml'
                mycopyfile('VOC%s/Annotations/%s' % (year, xmlname),'VOC%s/Annotations_Copy/%s' % (year, xmlname))
                convert_xml(year,xmlname)
            if line[12:14] == ' 1' and year == "2012":
                xmlname = line[0:11]+'.xml'
                mycopyfile('VOC%s/Annotations/%s' % (year, xmlname),'VOC%s/Annotations_Copy/%s' % (year, xmlname))
                convert_xml(year,xmlname)

for year in year_sets:
    for image_set in image_sets:
        for classesname in class_sets:
            select_annotation(year, image_set,classesname)

for year in year_sets:
    write_xml("./VOC%s/Annotations_Copy/"%(year))