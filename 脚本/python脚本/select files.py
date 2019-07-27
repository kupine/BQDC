import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
from shutil import copyfile


#classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","couch","pottedplant","bed","diningtable","toilet","tv","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
classes = ["n02747177","n02769748","n02795169","n02799071","n02802426","n02909870","n02917067","n02939185","n02971356","n02966687","n03014705","n03127925","n03481172","n03482405","n03743016","n03764736","n03976657","n04049303","n04204238","n04208210","n04254680","n04310018","n04311174","n04335435","n04367480","n04409515","n04487081","n04507155","n04540053"]
#classes = ["sportsball"]


wd = getcwd()


for classname in classes:
    newjpgpath='%s/newJPEGImages/%s/'%(wd,classname)
    if not os.path.exists(newjpgpath):
        os.makedirs(newjpgpath)

    label=search_xml('Annotations/%s'%classname)
    sorted(label)

    for image_id in label:
        copyfile('JPEGImages/%s/%s.JPEG'%(classname,image_id), 'newJPEGImages/classname/%s.jpg'%(image_set,classname,image_id))