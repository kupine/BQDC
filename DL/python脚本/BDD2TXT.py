# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:40:01 2018
�����������ݼ��Ľű�
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
ע�⣺����x1,y1��ʾ���ο����Ͻ����꣬x2,y2��ʾ���ο����½����꣬YOLOV3ʹ����Ҫת����x,y,w,h����Ϣд��
x,yΪ���ο����ĵ����꣬wΪ���ο��hΪ���ο��