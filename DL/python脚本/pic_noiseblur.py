# coding: utf-8
import numpy as np
import cv2
import random
from PIL import Image
import os

'''
    实现功能:
    对原始图片(xx.jpg)进行随机扩大(随机倍数区间见程序注释),生成a1_xx.jpg图片,同时生成新的xml标注文件
    对原始图片(xx.jpg)进行运动模糊处理,并加入噪点,生成a2_xx.jpg图片,同时生成新的xml标注文件
    对运动模糊处理,并加入噪点的图片(a2_xx.jpg)进行随机扩大(随机倍数区间见程序注释),生成a3_xx.jpg图片,同时生成新的xml标注文件
    
    扩大图片规则:
    将BACKGROUND_PIC_NAME图片扩大到相应倍数后,将原始图片放在此图片中间位置
'''

'''
    以下常量定义转换后的图片前缀名
'''
#随机扩大的图片文件名前缀
PREFIX_RESIZE = "a1_"
#原始图片处理后的文件名前缀
PREFIX_ORG_DEAL = "a2_"
#随机扩大图片处理后的文件名前缀
PREFIX_RESIZE_DEAL = "a3_"

'''
    以下常量定义添加椒盐噪声是否启用
    0:禁用 1:启用
'''
SP_NOISE_EN = 0

'''
    以下常量定义各种路径,需根据实际情况修改,windows系统下使用反斜杠"\\",类linux系统下使用正斜杠"//"
'''
#操作系统定义
IS_WINDOWS = 1
#背景图片文件名
BACKGROUND_PIC_NAME = "E:\\dealPIC\\background.jpg"
#图片路径定义
PIC_PATH = "E:\\dealPIC\\NCAC5\\JPEGImages\\train\\"
#XML标注文件路径定义
XML_PATH = "E:\\dealPIC\\NCAC5\\Annotations\\train\\"

if (IS_WINDOWS):
    PIC_PATH = PIC_PATH.replace("/","\\")
    XML_PATH = XML_PATH.replace("/","\\")
    BACKGROUND_PIC_NAME = BACKGROUND_PIC_NAME.replace("/","\\")
else:
    PIC_PATH = PIC_PATH.replace("\\", "/")
    XML_PATH = XML_PATH.replace("\\", "/")
    BACKGROUND_PIC_NAME = BACKGROUND_PIC_NAME.replace("\\", "/")

'''
    WALK_MODE定义了两种目录遍历模式
    0：只遍历PIC_PATH,XML_PATH下的文件,子目录不处理。图片文件应直接存放在PIC_PATH目录下,标注文件应直接存放在XML_PATH下
    1：遍历PIC_PATH下的子目录,PIC_PATH下的子目录应与XML_PATH子目录名称相同。如aa.jpg存放在PIC_PATH+"aa\\"子目录下,那么该图片对应的标注文件应存放在XML_PATH+"aa\\"子目录下
'''
#目录遍历模式
WALK_MODE = 1

#图片最大像素设置,超过此像素就不做扩大处理
MAX_PIX = 5000

#运动模糊1 输入:image
def motion_blur1(image, degree=20, angle=45):
    im= np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(im, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    if (SP_NOISE_EN == 1):
        return addsalt_pepper(blurred,2)

    return blurred
    #return blurred

#运动模糊2 输入:numpy array
def motion_blur2(im, degree=20, angle=45):
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(im, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


#图片添加椒盐噪声
def addsalt_pepper(img, NS):
    '''
    :param
    image: numpy.array, shape = [H, W, C]
    :param
    NS: noise
    strength(NS > 0)
    :return: numpy.array, shape = [H, W, C]
    '''
    assert NS > 0
    #ny_image = np.array(img, dtype=np.uint8)
    ny_image = img
    noise_mask = np.random.normal(0, 1,
                             size=(img.shape[0], img.shape[1]))
    for i in range(ny_image.shape[2]):
        ny_image[:,:,i] = (noise_mask >= NS) * 255 +\
                          (noise_mask <= (-NS)) * 0 +\
                          ((noise_mask > (-NS)) & (noise_mask < NS)) * ny_image[:,:,i]
    ny_image = ny_image.astype(np.uint8)
    #return ny_image
    return ny_image

'''
    图片扩大
    [input]
    orgfilename:    原始文件名(带全路径)
    dstfilename:    扩大后的图片文件名(带全路径)
    backgroundfile: 背景图片文件名(带全路径)
    [output]
    new_W:          扩大后的图片width
    new_H:          扩大后的图片height
    offset_W:       原始图片在新图片中的width方向偏移
    offset_H:       原始图片在新图片中的height方向偏移
'''
def ImageResize(orgfilename, dstfilename, backgroundfile):
    im_org = Image.open(orgfilename)
    im_back = Image.open(backgroundfile)

    org_W = im_org.size[0]
    org_H = im_org.size[1]


    tmp_TE = (int)(MAX_PIX / org_W)
    if (tmp_TE < 2):
        del im_org
        del im_back
        return 0,0,0,0,0
    tmp_INT = tmp_TE
    tmp_TE = (int)(MAX_PIX / org_H)
    if (tmp_TE < 2):
        del im_org
        del im_back
        return 0,0,0,0,0
    if (tmp_TE > tmp_INT):
        tmp_TE = tmp_INT

    box = (0,0,org_W,org_H)
    region = im_org.crop(box)

    if (tmp_TE > 10):
        tmp_TE = 10
    if (tmp_TE < 3):
        TB = 12
        TE = tmp_TE * 10
        TSTEP = 2
    elif (tmp_TE < 8):
        TB = 20
        TE = tmp_TE * 10
        TSTEP = 5
    else:
        TB = 20
        TE = tmp_TE * 10
        TSTEP = 10

    # 图片尺寸扩大比例 从TB起始 TE结束 步进量TSTEP
    fscale = random.randrange(TB, TE, TSTEP) / 10

    #背景图片放大
    new_W = (int)(org_W * fscale)
    new_H = (int)(org_H * fscale)

    #将背景图片变为正方形
    if (new_H > new_W):
        new_W = new_H
    else:
        new_H = new_W
    im_back = im_back.resize((new_W , new_H))

    #计算偏移量
    offset_H = (int)((new_H - org_H) / 2)
    offset_W = (int)((new_W - org_W) / 2)

    #region = region.transpose(Image.ROTATE_180)
    #计算原始图片在背景图片正中的box
    box_new = (offset_W, offset_H, org_W + offset_W, org_H + offset_H)
    im_back.paste(region, box_new)
    im_back.save(dstfilename)

    del region
    del im_org
    del im_back

    return 1,new_W, new_H, offset_W, offset_H

'''
    生成新图片对应的xml标注文件
    [input]
    orgfile:    原始文件名(带全路径)
    newfile:    新文件名(带全路径)
    picfile:    图片文件名(不带路径)
    wsize:      图片宽度
    hsize:      图片高度
    woffset:    宽度方向标注的偏移量
    hoffset:    高度方向标注的偏移量
'''
def CreateXmlFile(orgfile, newfile, picfile, wsize = 0, hsize = 0, woffset = 0, hoffset = 0):
    of = open(orgfile, "r", encoding='UTF-8')
    df = open(newfile, "w", encoding='UTF-8')

    line_item = of.readline()
    while (len(line_item) > 0):
        if ((wsize > 0) and (line_item.find("<width>") >= 0)):
            new_item = "<width>" + str(wsize)
            new_item += "</width>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif ((hsize > 0) and (line_item.find("<height>") >= 0)):
            new_item = "<height>" + str(hsize)
            new_item += "</height>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif (line_item.find("<filename>") >= 0):
            new_item = "<filename>" + picfile
            new_item += "</filename>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif ((woffset > 0) and (line_item.find("<xmin>") >= 0)):
            bgpos = line_item.find(">")
            edpos = line_item.rfind("<")
            strtmp = line_item[bgpos + 1:edpos]
            strtmp.strip(" ")
            new_item = "<xmin>" + str(int(strtmp) + woffset)
            new_item += "</xmin>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif ((woffset > 0) and (line_item.find("<xmax>") >= 0)):
            bgpos = line_item.find(">")
            edpos = line_item.rfind("<")
            strtmp = line_item[bgpos + 1:edpos]
            strtmp.strip(" ")
            new_item = "<xmax>" + str(int(strtmp) + woffset)
            new_item += "</xmax>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif ((woffset > 0) and (line_item.find("<ymin>") >= 0)):
            bgpos = line_item.find(">")
            edpos = line_item.rfind("<")
            strtmp = line_item[bgpos + 1:edpos]
            strtmp.strip(" ")
            new_item = "<ymin>" + str(int(strtmp) + hoffset)
            new_item += "</ymin>"
            new_item += '\x0a'
            df.writelines(new_item)
        elif ((woffset > 0) and (line_item.find("<ymax>") >= 0)):
            bgpos = line_item.find(">")
            edpos = line_item.rfind("<")
            strtmp = line_item[bgpos + 1:edpos]
            strtmp.strip(" ")
            new_item = "<ymax>" + str(int(strtmp) + hoffset)
            new_item += "</ymax>"
            new_item += '\x0a'
            df.writelines(new_item)
        else:
            line_item = line_item.replace("\r", "")
            df.write(line_item)
        line_item = of.readline()
    of.close()
    df.close()
    del of
    del df
    del line_item

'''
    判断filename是否是原始图片文件
    [input]
    filename:   图片文件名(不带路径)
    [output]
    iret:   0-不是原始图片 1-是原始图片
    fname:  文件名称去除扩展名
    exname: 扩展名
'''
def IsOrginFileName(filename):
    iret = 1

    fname = ""
    exname = ""

    if (filename.find(PREFIX_RESIZE) == 0):
        iret = 0
    elif (filename.find(PREFIX_ORG_DEAL) == 0):
        iret = 0
    elif (filename.find(PREFIX_RESIZE_DEAL) == 0):
        iret = 0
    if (iret == 0):
        return iret,fname,exname

    strend = filename.rfind(".")
    fname = filename[0:strend]
    strbegin = filename.rfind(".")
    exname = filename[strbegin:]

    return iret,fname,exname

# 通过给定的图片文件目录返回对应的xml标注文件所在目录
def GetXmlFilePath(pic_path):
    xmlpath = XML_PATH
    if (WALK_MODE == 1):
        if (IS_WINDOWS):
            keyword = "\\"
        else:
            keyword = "/"

        xmlpath = pic_path.rstrip(keyword)
        xmlpath = xmlpath[xmlpath.rfind(keyword) + 1:]
        xmlpath = XML_PATH + xmlpath + keyword

    return xmlpath

#图像添加椒盐噪声及运动模糊
def ImgNoiseBlur(orgfilename, dstfilename):
    tpimg = cv2.imread(orgfilename)
    tpimg_ = motion_blur1(tpimg, 5,90)
    cv2.imwrite(dstfilename, tpimg_)

    del tpimg
    del tpimg_

    return

iCounter = 0
# 遍历图片文件目录,对每个合法的图片文件进行处理
for root, dirs, files in os.walk(PIC_PATH, topdown=False):
    for name in files:
        #dirname = os.path.join(root, name)
        iret,fname,exname = IsOrginFileName(name)
        if (iret == 1):
            #原始图片全路径名
            pic_pathname_org = os.path.join(root, name)

            iCounter = iCounter + 1
            print("deal with No.%s:%s\n"%(iCounter,pic_pathname_org))

            # 1.原始图片放大后居中
            pic_pathname_resize = os.path.join(root, PREFIX_RESIZE)
            pic_pathname_resize = pic_pathname_resize + name
            done,new_W,new_H,offset_W,offset_H = ImageResize(pic_pathname_org, pic_pathname_resize, BACKGROUND_PIC_NAME) #调整大小

            if (done == 1):
                xml_pathname_org = GetXmlFilePath(root) + fname + ".xml"
                xml_pathname_dst = GetXmlFilePath(root) + PREFIX_RESIZE + fname + ".xml"
                CreateXmlFile(xml_pathname_org, xml_pathname_dst, PREFIX_RESIZE + name,new_W,new_H,offset_W,offset_H) #生成xml标注文件

            #2.原始图片加入噪点,并进行运动模糊处理
            pic_pathname_org_noiseblur = os.path.join(root, PREFIX_ORG_DEAL)
            pic_pathname_org_noiseblur = pic_pathname_org_noiseblur + name
            ImgNoiseBlur(pic_pathname_org, pic_pathname_org_noiseblur) #加入噪声和运动模糊
            xml_pathname_org = GetXmlFilePath(root) + fname + ".xml"
            xml_pathname_dst = GetXmlFilePath(root) + PREFIX_ORG_DEAL + fname + ".xml"
            CreateXmlFile(xml_pathname_org, xml_pathname_dst, PREFIX_ORG_DEAL + name)  # 生成xml标注文件

            #3.加入噪点和运动模糊的图片放大
            if (done == 1):
                pic_pathname_resize_noiseblur = os.path.join(root, PREFIX_RESIZE_DEAL)
                pic_pathname_resize_noiseblur = pic_pathname_resize_noiseblur + name
                done2,new_W, new_H, offset_W, offset_H = ImageResize(pic_pathname_org_noiseblur, pic_pathname_resize_noiseblur,BACKGROUND_PIC_NAME)  # 调整大小
                if (done2 == 1):
                    xml_pathname_org = GetXmlFilePath(root) + PREFIX_ORG_DEAL + fname + ".xml"
                    xml_pathname_dst = GetXmlFilePath(root) + PREFIX_RESIZE_DEAL + fname + ".xml"
                    CreateXmlFile(xml_pathname_org, xml_pathname_dst, PREFIX_RESIZE + name, new_W, new_H, offset_W,offset_H)  # 生成xml标注文件

