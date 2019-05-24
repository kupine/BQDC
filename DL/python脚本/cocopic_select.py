
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 10:46
# @Author  : He Hangjiang
# @Site    : 
# @Software: PyCharm
 
import json
import os,shutil
 
nameStr = ['person','dog']#填写需要筛选的分类
 
with open("D:/coco_copy/annotations_trainval2017/annotations/instances_val2017.json","r+") as f:
    data = json.load(f)
    print("read ready")
 
for i in data:
    imgName = "000000" + str(i["filename"]) + ".jpg"
    nameStr.append(imgName)
 
nameStr = set(nameStr)
print(len(nameStr))
 
path = "D:/coco/val2017/val2017/"
 
for file in os.listdir(path):
    if(file in nameStr):
        print("D:/coco/train2017/val2017/%s"%file)
        shutil.copy(path+file,"D:/coco_copy/val2017/val2017/"+file)