#coding=utf-8
from PIL import Image#需要pillow库
import os
import glob
from PIL import ImageFile

wd = os.getcwd()

njpgpath='%s/nJPEGImages/train/suitcase/'%(wd)
if not os.path.exists(njpgpath):
    os.makedirs(njpgpath)

ImageFile.LOAD_TRUNCATED_IMAGES = True
#in_dir = os.getcwd()
in_dir = '%s/JPEGImages/train/suitcase/'%(wd)#当前目录
#out_dir = in_dir+' mini'
out_dir = njpgpath#转换后图片目录
##percent = 0.4#缩放比例
#percent = input('请输入缩放比例：')
proportion = 0.1
proportion = float(proportion)
if not os.path.exists(out_dir): os.mkdir(out_dir)
#图片批处理
def main():
    for files in glob.glob(in_dir+'/*.jpg'):
        filepath,filename = os.path.split(files)
        im = Image.open(files)
        w,h = im.size
        im = im.resize((int(w*proportion), int(h*proportion)))
        im.save(os.path.join(out_dir,filename))
 
if __name__=='__main__':
    main()
print('you succeed.')