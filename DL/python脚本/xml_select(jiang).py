#jiangyiliang修改的筛选分类的脚本
import xml.etree.ElementTree as xml_tree
import os
import shutil

 
fileDir_ann = "./VOC2012/Annotations"
fileDir_img = "./VOC2012/JPEGImages/"
saveDir_img = "./VOC2012/JPEGImages_ssd/"
 
if not os.path.exists(saveDir_img):
    os.mkdir(saveDir_img)
 
#names = locals()
 
for files in os.walk(fileDir_ann):
    for file in files[2]:
        
        print(file + "-->start!")
 
        saveDir_ann = "./VOC2012/Annotations_ssd/"
        if not os.path.exists(saveDir_ann):
            os.mkdir(saveDir_ann)
 
        fp = open(fileDir_ann + '/' + file)        
        saveDir_ann = saveDir_ann + file
        fp_w = open(saveDir_ann, 'w')
       # classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',\
        #           'dog','horse','motorbike','pottedplant','sheep','sofa','train','tvmonitor','person']
        classes = ['person','train']
 
        lines = fp.readlines()
 
        ind_start = []
        ind_end = []
        lines_id_start = lines[:]
        lines_id_end = lines[:]

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
            fp_w.write(string_start[c])
        fp_w.close()
 
        if a == 0:
            os.remove(saveDir_ann)
        else:
            name_img = fileDir_img + os.path.splitext(file)[0] + ".jpg"
            shutil.copy(name_img,saveDir_img)
        fp.close()

