#jiangyiliang写的筛选指定分类的图片的脚本
import pickle
import os,shutil
from os import listdir, getcwd
from os.path import join

year_sets=["2012","2007"]
image_sets=["train","val"]
#class_sets = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_sets = ["person", "train"]

count = 0

def mycopyfile(srcfile,dstfile):
	if not os.path.isfile(srcfile):
		print "%s not exist!"%(srcfile)
	else:
		fpath,fname=os.path.split(dstfile)    
		if not os.path.exists(fpath):
			os.makedirs(fpath)               
		shutil.copyfile(srcfile,dstfile)     
		print "copy %s -> %s for %s_%s in %s"%(srcfile,dstfile,classesname,image_set,year)


def select_annotation(year, image_set,classesname):
	filename = 'VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year, classesname,image_set)
	with open(filename) as file_object:
		lines = file_object.readlines()
		for line in lines:
			if line[7:9] == ' 1' and year == "2007":
				imagename = line[0:6]+'.jpg'
				mycopyfile('VOCdevkit/VOC%s/JPEGImages/%s'%(year,imagename),'VOCdevkit/VOC%s/JPEGImages_Copy/%s'%(year,imagename))
			if line[12:14] == ' 1' and year == "2012":
				imagename = line[0:11]+'.jpg'
				mycopyfile('VOCdevkit/VOC%s/JPEGImages/%s'%(year,imagename),'VOCdevkit/VOC%s/JPEGImages_Copy/%s'%(year,imagename))	


for year in year_sets:
    for image_set in image_sets:
		for classesname in class_sets:
			select_annotation(year, image_set,classesname)
