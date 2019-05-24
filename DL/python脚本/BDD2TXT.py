# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:40:01 2018
处理伯克利数据集的脚本
@author: phinoo
"""
import json
categorys=['car','person']
jsonFile="./data/bdd100k_labels_images_val.json"

f=open(jsonFile)
info=json.load(f)

for image_index in range(0,len(info)):
    strs=""
    image=info[image_index]
    write=open("./result/%s.txt"%(image["name"][0:-4]),'w')
    for labels_index in range(0,len(image["labels"])):
        
        labels=image["labels"][labels_index]
        if labels["category"] in categorys:
            strs+=labels["category"]
            strs+=" "
            strs+=str((labels["box2d"]["x1"]/1280))
            strs+=" "
            strs+=str((labels["box2d"]["y1"]/720))
            strs+=" "
            strs+=str((labels["box2d"]["x2"]/1280))
            strs+=" "
            strs+=str((labels["box2d"]["y2"]/720))
            strs+="\n"
    write.writelines(strs)
    write.close()
    print("%s has been deal!"%image["name"])
注意：上述x1,y1表示矩形框左上角坐标，x2,y2表示矩形框右下角坐标，YOLOV3使用需要转化成x,y,w,h的信息写入
x,y为矩形框中心点坐标，w为矩形框宽，h为矩形框高