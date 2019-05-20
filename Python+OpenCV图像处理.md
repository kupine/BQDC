一、读取显示一张图片：

配置好所有环境后，开始利用python+opencv进行图像处理第一步。

读取和显示一张图片：

```
import cv2 as cv
src=cv.imread('E:\imageload\example.png')       
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', src)
cv.waitKey(0)
cv.destroyAllWindows()
```

输出效果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180505001231309-1256247491.png)

代码解释：

　　src=cv.imread('E:\imageload\example.png')   

　　#读取这个路径的图片     注意这里的路径必须全是英文，不能有中文       但是分隔符\是随意的，还可以是 /   \\    // 形式的 （在python3至少是这样） 

　　cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) 

　　 #namedWindow函数，用于创建一个窗口        默认值为WINDOW_AUTOSIZE，所以一般情况下，这个函数我们填第一个变量就可以了。其实这一行代码没有也可以正常显示的（下面imshow会显示）

　　cv.imshow('input_image', src) 

　　#在指定的窗口中显示一幅图像

　　cv.waitKey(0)           

　　#   参数=0: （也可以是小于0的数值）一直显示，不会有返回值      若在键盘上按下一个键即会消失 ，则会返回一个按键对应的ascii码值       

　　　 参数>0:显示多少毫秒        超过这个指定时间则返回-1 

　　cv.destroyAllWindows()  

　　 *#*删除建立的全部窗口，释放资源 

注意：若同时使用namedWindow和imshow函数，则两个函数的第一个参数名字必须相同。

重要一点：在pycahrm里一定要把Project Encoding设置为utf-8，否则在新建的py文件里注释中文字符时，Pycharm运行会报错。

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180505142304732-1055147163.png)

当然如果已经新建了py文件，要避免报错的话，应该在代码第一行注释：#encoding=gbk



二、打印图片属性、设置图片存储路径、电脑摄像头的调取和显示

2.1、 打印图片属性、设置图片存储路径

代码如下：

```
#打印图片的属性、保存图片位置
import cv2 as cv
import numpy as np     #numpy是一个开源的Python科学计算库
def get_image_info(image):
    print(type(image))   #type() 函数如果只有第一个参数则返回对象的类型   在这里函数显示图片类型为 numpy类型的数组
    print(image.shape)
    #图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，
    # 第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成
    print(image.size)  #返回图像的大小，size的具体值为shape三个元素的乘积
    print(image.dtype) #数组元素的类型通过dtype属性获得
    pixel_data=np.array(image)
    print(pixel_data) # 打印图片矩阵     N维数组对象即矩阵对象
src=cv.imread('E:\imageload\example.png')
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', src)
get_image_info(src)
cv.imwrite("E:/example.png",src)       #图片存储路径
# gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)   #使图片颜色变为灰度
# cv.imwrite("E:/example.png",gray)
cv.waitKey(0)
cv.destroyAllWindows()
```

2.2、电脑摄像头的调取和显示

代码如下：

```
#电脑摄像头的调取和显示
import cv2 as cv
def video_demo():
    capture=cv.VideoCapture(0)
    #参数为视频设备的id ，如果只有一个摄像头可以填0，表示打开默认的摄像头     这里的参数也可以是视频文件名路径，只要把视频文件的具体路径写进去就好
    while True:  #只要没跳出循环，则会循环播放每一帧 ,waitKey(10)表示间隔10ms
        ret, frame = capture.read()
        #read函数读取视频(摄像头)的某帧,它能返回两个参数. 第一个参数是bool型的ret，其值为True或False，代表有没有读到图片. 第二个参数是frame，是当前截取一帧的图片
        frame=cv.flip(frame,1)
        #翻转  0:沿X轴翻转(垂直翻转)   大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        cv.imshow("video",frame)
        pc=cv.waitKey(10)   #超过10ms, waitKey函数会返回-1，如果10ms内在键盘按了某个按键，则 waitKey函数会返回对应按键的ASCII码值，ASCII码值一定大于0
        if pc>0:
            break
        # if cv.waitKey(10) == ord('z'):  # 键盘输入z退出窗口，不按z点击关闭会一直关不掉 也可以设置成其他键。 ord()函数返回对应字符的ASCII数值
        #     break
video_demo()
cv.destroyAllWindows()
```

三、Numpy数组操作图片

3.1、改变图片每个像素点每个通道的灰度值

方法一代码如下：

```
#遍历访问图片每个像素点，并修改相应的RGB
import cv2 as cv
def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width: %s  height: %s  channels: %s"%(width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row , col, c]        #获取每个像素点的每个通道的数值
                image[row, col, c]=255 - pv     #灰度值是0-255   这里是修改每个像素点每个通道灰度值
    cv.imshow("second_image",image)
src=cv.imread('E:\imageload\example.png')   #blue, green, red
cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE)
cv.imshow('first_image', src)
t1 = cv.getTickCount()    #GetTickcount函数返回从操作系统启动到当前所经过的毫秒数
access_pixels(src)
t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()  #getTickFrequency函数返回CPU的频率,就是每秒的计时周期数
print("time : %s ms"%(time*1000) )    #输出运行时间
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506150623140-85954947.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506151222079-690544940.png)

 

注意：

1.image[i,j,c]   i表示图片的行数，j表示图片的列数，c表示图片的通道数（0代表B，1代表G，2代表R    一共是RGB三通道）。坐标是从左上角开始

2.每个通道对应一个灰度值。灰度值概念：把白色与黑色之间按对数关系分成若干级，称为“灰度等级”。范围一般从0到255，白色为255，黑色为0。要详细了解灰度值和通道的概念，请参考这篇博客：<https://blog.csdn.net/silence2015/article/details/53789748>



方法二，上述代码实现像素取反的运行时间较长，下面代码运用opencv自带的库函数可以使运行时间大大减少，代码如下：

```
#调用opencv的库函数快速实现像素取反
import cv2 as cv
def inverse(img):
    img = cv.bitwise_not(img)   #函数cv.bitwise_not可以实现像素点各通道值取反
    cv.imshow("second_image", img)

src=cv.imread('E:\imageload\example.png')   #blue, green, red
cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE)
cv.imshow('first_image', src)
t1 = cv.getTickCount()    #GetTickcount函数返回从操作系统启动到当前所经过的毫秒数
inverse(src)
t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()  #getTickFrequency函数返回CPU的频率,就是每秒的计时周期数
print("time : %s ms"%(time*1000) )    #输出运行时间
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506214110199-1305370002.png)

可见，使用库函数 bitwise_not 可以使运行时间缩短13倍左右

3.2、自定义一张三通道图片

代码如下：

```
#自定义一张三通道图片
import cv2 as cv
import numpy as np
def creat_image():
    img = np.zeros([400, 400, 3], np.uint8)   #将所有像素点的各通道数值赋0
    img[:, :, 0] = np.ones([400, 400]) * 255   #0通道代表B
    # img[:, :, 1] = np.ones([400, 400]) * 255   #1通道代表G
    # img[:, :, 2] = np.ones([400, 400]) * 255   #2通道代表R
    cv.imshow("new_image",img)
creat_image()
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：
![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506162417994-1099646616.png)

 注意：

1.np.zeros函数用于创建一个数值全为0的矩阵，np.ones用于创建一个数值全为1的矩阵

2.当图片为多通道图片时，B:255  G:0  R:0 则三通道图片显示蓝色。所有通道数值组合示意图如下：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506163756302-1382904033.jpg)

 补注：

单通道： 此通道上值为0－255。 （255为白色，0是黑色） 只能表示灰度，不能表示彩色。
三通道：BGR （255，255，255为白色， 0,0,0是黑色 ）  可以表示彩色， 灰度也是彩色的一种。

单通道和三通道区别见博客：<https://blog.csdn.net/qq_32211827/article/details/56854985>

3.3、自定义一张单通道图片

代码如下：

```
#自定义一张单通道图片
import cv2 as cv
import numpy as np
def creat_image():
    img = np.ones([400, 400, 1], np.uint8)   #该像素点只有一个通道，该函数使所有像素点的通道的灰度值为1
    img = img * 127       #使每个像素点单通道的灰度值变为127
    cv.imshow("new_image",img)
creat_image()
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180506194102354-1764325042.png)

注意：

1.代码里 img = img * 127    表示数组里的每个数值都乘以127

2.之所以np.ones函数参数类型是uint8，是因为uint8数的范围为0~255,  那么为0时恰好为黑色，为255时恰好为白色。若函数参数类型为int8，则int8类型数的范围为-128~127，那么-128则为黑色，127为白色。

四、色彩空间

4.1、色彩空间的转换

代码如下：

```
#色彩空间转换
import cv2 as cv
def color_space_demo(img):
    gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #RGB转换为GRAY #这里的生成的gray图是单通道的
 cv.imshow("gray", gray) hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #RGB转换为HSV cv.imshow("hsv", hsv) yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV) #RGB转换为YUV cv.imshow("yuv",yuv) Ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb) #RGB转换为YCrCb cv.imshow("Ycrcb", Ycrcb) src = cv.imread('E:\imageload\example.png') cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE) cv.imshow('first_image', src) color_space_demo(src) cv.waitKey(0) cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180507141008030-660336884.png)

 

注意：参考博文：<https://blog.csdn.net/a352611/article/details/51416769>

1.RGB就是指Red,Green和Blue,一副图像由这三个channel(通道)构成

2.Gray就是只有灰度值一个channel。 

3.HSV即Hue(色调),Saturation(饱和度)和Value(亮度)三个channel

 

切记（纯属个人理解）：

1.百度百科说，将原来的RGB(R,G,B)中的R,G,B统一按照一种转换关系用Gray替换，形成新的颜色RGB(Gray,Gray,Gray)，用它替换原来的RGB(R,G,B)就是灰度图。

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513212807364-1349796220.png)

2.opencv里，COLOR_RGB2GRAY是将三通道RGB对象转换为单通道的灰度对象。

3.将单通道灰度对象转换为 RGB 时，生成的RGB对象的每个通道的值是灰度对象的灰度值。

 

 

RGB是为了让机器更好的显示图像,对于人类来说并不直观,HSV更为贴近我们的认知,所以通常我们在针对某种颜色做提取时会转换到HSV颜色空间里面来处理. 

补注：

1.HSV如下图：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180507141411808-632910687.png)

2.opencv里HSV色彩空间范围为： H：0-180  S: 0-255   V： 0-255

3.常见的色彩空间有RGB、HSV、HIS、YCrCb、YUV，其中最常用的是RGB、HSV、YUV，其中YUV就是YCrCb(详见百度百科)。其中YUV的“Y”表示明亮度（Luminance或Luma），也就是灰阶值；而“U”和“V”  表示的则是色度（Chrominance或Chroma），作用是描述影像色彩及饱和度，用于指定像素的颜色。

4.2、利用inrange函数过滤视频中的颜色，实现对特定颜色的追踪

代码如下：

```
#视频特定颜色追踪
import cv2 as cv
import numpy as np
def extrace_object_demo():
    capture=cv.VideoCapture("E:/imageload/video_example.mp4")
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)      #色彩空间由RGB转换为HSV
        lower_hsv = np.array([100, 43, 46])             #设置要过滤颜色的最小值
        upper_hsv = np.array([124, 255, 255])           #设置要过滤颜色的最大值
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)    #调节图像颜色信息（H）、饱和度（S）、亮度（V）区间，选择蓝色区域
        cv.imshow("video",frame)
        cv.imshow("mask", mask)
        c = cv.waitKey(40)
        if c == 27:      #按键Esc的ASCII码为27
            break
extrace_object_demo()
cv.destroyAllWindows()
```

运行结果：

这里只放追踪蓝色部分的截图，仅供参考

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180507154901974-1331646272.png)

注意：

1.Opencv的inRange函数：可实现二值化功能

函数原型：inRange(src,lowerb, upperb[, dst]) -> dst         

　　函数的参数意义：第一个参数为原数组，可以为单通道，多通道。第二个参数为下界，第三个参数为上界

例如：mask = cv2.inRange(hsv, lower_blue, upper_blue)      

　　第一个参数：hsv指的是原图（原始图像矩阵）

　　第二个参数：lower_blue指的是图像中低于这个lower_blue的值，图像值变为255

　　第三个参数：upper_blue指的是图像中高于这个upper_blue的值，图像值变为255 （255即代表黑色）

　　而在lower_blue～upper_blue之间的值变成0 (0代表白色)

即：Opencv的inRange函数可提取特定颜色，使特定颜色变为白色，其他颜色变为黑色，这样就实现了二值化功能

2.HSV颜色对应的RGB分量范围表如下：（这里是三通道的）

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180507162133947-829385003.png)

4.3、通道的分离、合并以及某个通道值的修改

代码如下：

```
#通道的分离与合并以及某个通道值的修改
import cv2 as cv
src=cv.imread('E:\imageload\example.png')
cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE)
cv.imshow('first_image', src)

#三通道分离形成单通道图片
b, g, r =cv.split(src)
cv.imshow("second_blue", b)
cv.imshow("second_green", g)
cv.imshow("second_red", r)
# 其中cv.imshow("second_red", r)可表示为r = cv2.split(src)[2]

#三个单通道合成一个三通道图片
src = cv.merge([b, g, r])
cv.imshow('changed_image', src)

#修改多通道里的某个通道的值
src[:, :, 2] = 0
cv.imshow('modify_image', src)

cv.waitKey(0)
cv.destroyAllWindows()
```

注意：

1.这里用到了opencv的split函数和merge函数，实现通道的分离和合并。

2.cv.split函数分离出的b、g、r是单通道图像。

五、像素运算

5.1、像素的算术运算

像素的算术运算涉及加减乘除等基本运算（要进行算术运算，两张图片的形状（shape）必须一样）

代码如下：

```
#像素的算术运算(加、减、乘、除)   两张图片必须shape一致
import cv2 as cv
def add_demo(m1, m2):   #像素的加运算
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)
def subtract_demo(m1, m2):   #像素的减运算
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)
def divide_demo(m1, m2):   #像素的除法运算
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)
def multiply_demo(m1, m2):   #像素的乘法运算    
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)

src1 = cv.imread('E:\imageload\LinuxLogo.jpg')
src2 = cv.imread('E:\imageload\WindowsLogo.jpg')
cv.imshow('image1', src1)
cv.imshow('image2', src2)

add_demo(src1, src2)
subtract_demo(src1, src2)
divide_demo(src1, src1)
multiply_demo(src1, src2)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513151752759-1406841736.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513151831345-46308716.png)

注意：

1.这里的的像素运算指的是多维数组对应的值进行加减乘除运算，前提是两张图片必须shape、size一样

2.在相除的时候，一个很小的数除以很大的数结果必然小，所以得出的图像几乎全黑。（黑色为0，白色为255）

3.在相乘的时候，图案“Linux”边缘上的像素并不稳定。

5.2、像素的逻辑运算

像素的逻辑运算涉及与、或、非、异或等基本运算（要进行逻辑运算，两张图片的形状（shape）必须一样）

这里主要展示与或非的逻辑运算

代码如下：

```
#像素的逻辑运算(与、或、非)   两张图片必须shape一致
import cv2 as cv
def and_demo(m1, m2):    #与运算  每个像素点每个通道的值按位与
    dst = cv.bitwise_and(m1, m2)
    cv.imshow("and_demo", dst)
def or_demo(m1, m2):     #或运算   每个像素点每个通道的值按位或
    dst = cv.bitwise_or(m1, m2)
    cv.imshow("or_demo", dst)
def not_demo(m1):     #非运算   每个像素点每个通道的值按位取反
    dst = cv.bitwise_not(m1)
    cv.imshow("not_demo", dst)

src1 = cv.imread('E:\imageload\LinuxLogo.jpg')
src2 = cv.imread('E:\imageload\WindowsLogo.jpg')
cv.imshow('image1', src1)
cv.imshow('image2', src2)

and_demo(src1, src2)
or_demo(src1, src2)
not_demo(src1)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513154543295-438206993.png)

 

注意：这里的逻辑运算是按照像素点的各通道的值按二进制形式按位与或非进行运算的。

5.3、调节图片对比度和亮度

代码如下：

```
#调节图片对比度和亮度
import cv2 as cv
import numpy as np
def contrast_brightness_image(img1, ratio, b):    #第2个参数rario为对比度  第3个参数b为亮度
    h, w, ch = img1.shape
    img2 = np.zeros([h, w, ch], img1.dtype)    # 新建的一张全黑图片和img1图片shape类型一样，元素类型也一样
    dst = cv.addWeighted(img1, ratio, img2, 1 - ratio, b)
    cv.imshow("csecond", dst)
src = cv.imread("E:\imageload\example.png")
cv.imshow("first", src)
contrast_brightness_image(src, 0.1, 10)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513195511981-1682411976.png)

注意：help(cv2.addWeighted)可得到.addWeighted函数的官方解释。

函数addWeighted的原型：addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst

src1表示需要加权的第一个数组（上述例子就是图像矩阵）

alpha表示第一个数组的权重

src2表示第二个数组（和第一个数组必须大小类型相同）

beta表示第二个数组的权重

gamma表示一个加到权重总和上的标量值

即输出后的图片矩阵：dst = src1*alpha + src2*beta + gamma;

六、ROI与泛洪填充

6.1、ROI

ROI（region of interest），感兴趣区域。机器视觉、图像处理中，从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域，称为感兴趣区域，ROI。

代码如下：

```
#进行图片截取、合并、填充
import cv2 as cv
src=cv.imread('E:\imageload\lena.jpg')
cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE)
cv.imshow('first_image', src)
face = src[200:300, 200:400]    #选择200:300行、200:400列区域作为截取对象
gray = cv.cvtColor(face, cv.COLOR_RGB2GRAY)  #生成的的灰度图是单通道图像
backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  #将单通道图像转换为三通道RGB灰度图，因为只有三通道的backface才可以赋给三通道的src
src[200:300, 200:400] = backface
cv.imshow("face", src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513215343581-881067683.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180513215401136-1435656318.png)

注意：COLOR_RGB2GRAY是把三通道RGB对象转换为单通道灰度对象。

6.2、泛洪填充（彩色图像填充

代码如下：

```
#泛洪填充(彩色图像填充)
import cv2 as cv
import numpy as np
def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    #为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    cv.floodFill(copyImg, mask, (220, 250), (0, 255, 255), (100, 100, 100), (50, 50 ,50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImg)

src = cv.imread('E:/imageload/baboon.jpg')
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', src)
fill_color_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180515132932003-536213723.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180515132953751-698221371.png)

注意：

1.opencv里的mask都是为uin8类型的单通道阵列

2.泛洪填充算法也叫漫水填充算法。opencv的floodFill函数原型： floodFill(image, mask,  seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image,  mask, rect

　　image参数表示输入/输出1或3通道，8位或浮点图像。

　　mask参数表示掩码，该掩码是单通道8位图像，比image的高度多2个像素，宽度多2个像素。填充时不能穿过输入掩码中的非零像素。

　　seedPoint参数表示泛洪算法(漫水填充算法)的起始点。

　　newVal参数表示在重绘区域像素的新值。

　　loDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之负差的最大值。

　　upDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之正差的最大值。

　　flags参数：操作标志符，包含三部分：（参考<https://www.cnblogs.com/little-monkey/p/7598529.html>）

　　　　低八位（0~7位）：用于控制算法的连通性，可取4（默认）或8。

　　　　中间八位（8~15位）：用于指定掩码图像的值，但是如果中间八位为0则掩码用1来填充。

　　　　高八位（16~32位）：可以为0或者如下两种标志符的组合：

　　　　FLOODFILL_FIXED_RANGE:表示此标志会考虑当前像素与种子像素之间的差，否则就考虑当前像素与相邻像素的差。FLOODFILL_MASK_ONLY:表示函数不会去填充改变原始图像,而是去填充掩码图像mask，mask指定的位置为零时才填充，不为零不填充。　　

3.个人理解：参数3起始点的像素值减去参数5的像素值表示的是从起始点开始搜索周边范围的像素最低值，参数3起始点的像素值加上参数5的像素值表示的是从起始点开始搜索周边范围的像素最大值。有了这个范围，然后该函数就可以在这个连续像素范围内填充指定的颜色newVal参数值。

4.设置FLOODFILL_FIXED_RANGE – 改变图像，泛洪填充     

   设置FLOODFILL_MASK_ONLY – 不改变图像，只填充遮罩层本身，忽略新的颜色值参数。

6.3、泛洪填充（二值图像填充）

代码如下：

```
#泛洪填充(二值图像填充)
import cv2 as cv
import numpy as np
def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300] = 255
    cv.imshow("fill_binary", image)
    mask = np.ones([402, 402], np.uint8)   #mask要保证比原图像高和宽都多2
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200,200), (255 , 0, 0), cv.FLOODFILL_MASK_ONLY) #mask不为0的区域不会被填充，mask为0的区域才会被填充
    cv.imshow("filled_binary", image)
fill_binary()
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180518002654756-320487157.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180518002706216-393751501.png)

注意：

1.个人认为，不管是FLOODFILL_FIXED_RANGE还是FLOODFILL_MASK_ONLY操作，泛洪填充都不会填充掩膜mask的非零像素区域

2. mask[101:301, 101:301] = 0  这条语句为什么是101:301而不是100:300呢？我觉得应该是掩膜mask是比原图像左右上下都多了1，所以掩膜mask左右一共比原图像多2，上下也比原图像多2。那么原图像的100就自然对应到掩膜的101，同样原图像的300就自然对应到掩膜的301。

3.当FLOODFILL_MASK_ONLY设置了的时候，原图不会改变，只会用中间八位的值填冲mask。 floodFill的flags参数的中间八位的值就是用于指定填充掩码图像的值的，但是如果flags中间八位的值为0，则掩码会用1来填充。

七、滤波与模糊操作

​		过滤是信号和图像处理中基本的任务。其目的是根据应用环境的不同，选择性的提取图像中某些认为是重要的信息。过滤可以移除图像中的噪音、提取感兴趣的可视特征、允许图像重采样等等。频域分析将图像分成从低频到高频的不同部分。低频对应图像强度变化小的区域，而高频是图像强度变化非常大的区域。在频率分析领域的框架中，滤波器是一个用来增强图像中某个波段或频率并阻塞（或降低）其他频率波段的操作。低通滤波器是消除图像中高频部分，但保留低频部分。高通滤波器消除低频部分。参考博客：<https://blog.csdn.net/sunny2038/article/details/9155893>

　　个人认为模糊操作就是过滤掉图像中的一些特殊噪音。

　　具体模糊和滤波的关系如下图：参考知乎大神：<https://www.zhihu.com/question/54918332/answer/142137732>

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180530121552228-1041027459.png)

7.1、均值模糊、中值模糊、用户自定义模糊

代码如下：

```
#均值模糊、中值模糊、自定义模糊    模糊是卷积的一种表象
import cv2 as cv
import numpy as np

def blur_demo(image):      #均值模糊  去随机噪声有很好的去燥效果
    dst = cv.blur(image, (1, 15))    #（1, 15）是垂直方向模糊，（15， 1）还水平方向模糊
    cv.namedWindow('blur_demo', cv.WINDOW_NORMAL)
    cv.imshow("blur_demo", dst)

def median_blur_demo(image):    # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv.medianBlur(image, 5)
    cv.namedWindow('median_blur_demo', cv.WINDOW_NORMAL)
    cv.imshow("median_blur_demo", dst)

def custom_blur_demo(image):    # 用户自定义模糊
    kernel = np.ones([5, 5], np.float32)/25   #除以25是防止数值溢出 
    dst = cv.filter2D(image, -1, kernel)
    cv.namedWindow('custom_blur_demo', cv.WINDOW_NORMAL)
    cv.imshow("custom_blur_demo", dst)

src = cv.imread('E:\imageload\lenanoise.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL)
cv.imshow('input_image', src)

blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180527205628119-300924878.png)

注意：

1.均值滤波是典型的[线性](https://baike.baidu.com/item/线性)滤波算法，它是指在图像上对目标像素给一个模板，该模板包括了其周围的临近像素（以目标像素为中心的周围8个像素，构成一个滤波模板，即去掉目标像素本身），再用模板中的全体像素的平均值来代替原来像素值。

低通滤波（均值模糊）函数原型：blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst

src参数表示待处理的输入图像。

ksize参数表示模糊内核大小。比如(1,15)表示生成的模糊内核是一个1*15的矩阵。

dst参数表示输出与src相同大小和类型的图像。

anchor参数、borderType参数可忽略

2.[中值滤波法](https://baike.baidu.com/item/中值滤波法)是一种非线性平滑技术，它将每一[像素](https://baike.baidu.com/item/像素)点的[灰度值](https://baike.baidu.com/item/灰度值)设置为该点某邻域窗口内的所有像素点灰度值的[中值](https://baike.baidu.com/item/中值)。具体原理参见博客：<https://blog.csdn.net/weixin_37720172/article/details/72627543>

中值滤波（中值模糊）函数原型：medianBlur(src, ksize[, dst]) -> dst

src参数表示待处理的输入图像。

ksize参数表示滤波窗口尺寸，必须是奇数并且大于1。比如这里是5，中值滤波器就会使用5×5的范围来计算，即对像素的中心值及其5×5邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。

dst参数表示输出与src相同大小和类型的图像。

3.用户自定义模糊

所用函数：filter2D()

函数原型： filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst

src参数表示待处理的输入图像。

ddepth参数表示目标图像深度，输入值为-1时，目标图像和原图像深度保持一致

 kernel: 卷积核（或者是相关核）,一个单通道浮点型矩阵。修改kernel矩阵即可实现不同的模糊。

7.2、高斯模糊

代码如下：

```
#高斯模糊    轮廓还在，保留图像的主要特征  高斯模糊比均值模糊去噪效果好
import cv2 as cv
import numpy as np

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):        #加高斯噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]   #blue
            g = image[row, col, 1]   #green
            r = image[row, col, 2]   #red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.namedWindow("noise image", cv.WINDOW_NORMAL)
    cv.imshow("noise image", image)
    dst = cv.GaussianBlur(image, (15, 15), 0)  # 高斯模糊
    cv.namedWindow("Gaussian", cv.WINDOW_NORMAL)
    cv.imshow("Gaussian", dst)

src = cv.imread('E:\imageload\lena.jpg')
cv.namedWindow("input_image", cv.WINDOW_NORMAL)
cv.imshow('input_image', src)

gaussian_noise(src)
dst = cv.GaussianBlur(src, (15,15), 0)   #高斯模糊
cv.namedWindow("Gaussian Blur", cv.WINDOW_NORMAL)
cv.imshow("Gaussian Blur", dst)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180530121038838-720385618.png)

注意：

​		1.高斯模糊实质上就是一种均值模糊，只是高斯模糊是按照加权平均的，距离越近的点权重越大，距离越远的点权重越小。通俗的讲，高斯滤波就是对整幅图像进行[加权平均](https://baike.baidu.com/item/加权平均)的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。

​		2.高斯分布的一维和二维原理如下：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180530123741270-2030114961.png)

 　　补：高斯分布的标准差σ。标准差代表着数据的离散程度，如果σ较小，那么生成的模板的中心系数较大，而周围的系数较小，这样对图像的平滑效果就不是很明显；反之，σ较大，则生成的模板的各个系数相差就不是很大，比较类似均值模板，对图像的平滑效果比较明显。高斯模糊具体原理见博文：<https://blog.csdn.net/u012992171/article/details/51023768>

​		3.高斯模糊GaussianBlur函数原型：GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst

src参数表示待处理的输入图像。

​		ksize参数表示高斯滤波器模板大小。 ksize.width和ksize.height可以不同，但它们都必须是正数和奇数。或者，它们可以是零，即（0, 0），然后从σ计算出来。

​		sigmaX参数表示 X方向上的高斯内核标准差。

​		sigmaY参数表示 Y方向上的高斯内核标准差。 如果sigmaY为零，则设置为等于sigmaX，如果两个sigma均为零，则分别从ksize.width和ksize.height计算得到。

　　补：若ksize不为(0, 0)，则按照ksize计算，后面的sigmaX没有意义。若ksize为(0, 0)，则根据后面的sigmaX计算ksize

​		4.numpy包里的random模块用于生成随机数，random模块里的normal函数表示的是生成高斯随机数。

​		normal函数默认原型：normal(loc=0.0, scale=1.0, size=None)。

​		loc参数表示高斯分布的中心点。

​		scale参数表示高斯分布的标准差σ。

​		size参数表示产生随机数的个数。size取值可以为（m，n，k），表示绘制m*n*k个样本。

7.3、边缘保留滤波EPF

进行边缘保留滤波通常用到两个方法：高斯双边滤波和均值迁移滤波。

代码如下：

```
#边缘保留滤波（EPF）  高斯双边、均值迁移
import cv2 as cv
import numpy as np

def bi_demo(image):   #双边滤波
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.namedWindow("bi_demo", cv.WINDOW_NORMAL)
    cv.imshow("bi_demo", dst)

def shift_demo(image):   #均值迁移
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.namedWindow("shift_demo", cv.WINDOW_NORMAL)
    cv.imshow("shift_demo", dst)

src = cv.imread('E:/imageload/example.png')
cv.namedWindow('input_image', cv.WINDOW_NORMAL)
cv.imshow('input_image', src)

bi_demo(src)
shift_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180530154753905-1303288617.png)

注意：

​		1.[双边滤波](https://baike.baidu.com/item/双边滤波)（Bilateral filter）是一种非线性的滤波方法，是结合图像的空间[邻近度](https://baike.baidu.com/item/邻近度/4830086)和像素值相似度的一种折中处理，同时考虑空域信息和[灰度](https://baike.baidu.com/item/灰度/4615393)相似性，达到保边去噪的目的。双边滤波器顾名思义比高斯滤波多了一个[高斯](https://baike.baidu.com/item/高斯/10149932)方差sigma－d，它是基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波

​		2.双边滤波函数原型：bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst

​		src参数表示待处理的输入图像。

​		d参数表示在过滤期间使用的每个像素邻域的直径。如果输入d非0，则sigmaSpace由d计算得出，如果sigmaColor没输入，则sigmaColor由sigmaSpace计算得出。

​		sigmaColor参数表示色彩空间的标准方差，一般尽可能大。较大的参数值意味着像素邻域内较远的颜色会混合在一起，从而产生更大面积的半相等颜色。

​		sigmaSpace参数表示坐标空间的标准方差(像素单位)，一般尽可能小。参数值越大意味着只要它们的颜色足够接近，越远的像素都会相互影响。当d > 0时，它指定邻域大小而不考虑sigmaSpace。 否则，d与sigmaSpace成正比。

​		双边滤波原理：

<https://blog.csdn.net/edogawachia/article/details/78837988>，<https://blog.csdn.net/MoFMan/article/details/77482794>   ，<https://www.cnblogs.com/qiqibaby/p/5296681.html>  反正我是没怎么看懂o(╥﹏╥)o

​		3.均值漂移pyrMeanShiftFiltering函数原型：pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst

​		src参数表示输入图像，8位，三通道图像。

​		sp参数表示漂移物理空间半径大小。

​		sr参数表示漂移色彩空间半径大小。

​		dst参数表示和源图象相同大小、相同格式的输出图象。

​		maxLevel参数表示金字塔的最大层数。

​		termcrit参数表示漂移迭代终止条件。

均值漂移原理：

<https://blog.csdn.net/dcrmg/article/details/52705087> 

<https://blog.csdn.net/qq_23968185/article/details/51804574>

<https://blog.csdn.net/jinshengtao/article/details/30258833>

八、图像直方图

​		直方图简介：图像的直方图是用来表现图像中亮度分布的直方图,给出的是图像中某个亮度或者某个范围亮度下共有几个像素.还不明白?就是统计一幅图某个亮度像素数量.比如对于灰度值12,一幅图里面有2000
个像素其灰度值为12,那么就能够统计12这个亮度的像素为2000个,其他类推。参考：<https://blog.csdn.net/xierhacker/article/details/52605308>

8.1、安装matplotlib

要画直方图必须要安装matplotlib库，Matplotlib 是一个 Python 的 2D绘图库。

安装步骤：

运行cmd，然后在自己的python安装路径的Scripts文件夹目录下，输入命令： pip install matplotlib

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180531184647333-1479401636.png)

8.2、画直方图

代码如下：

```
#画直方图
import cv2 as cv
from matplotlib import pyplot as plt

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])         #numpy的ravel函数功能是将多维数组降为一维数组
    plt.show()

def image_hist(image):     #画三通道图像的直方图
    color = ('b', 'g', 'r')   #这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i , color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])  #计算直方图
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()

src = cv.imread('E:/imageload/WindowsLogo.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL)
cv.imshow('input_image', src)

plot_demo(src)
image_hist(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180531202147977-721822822.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180531202209633-165082136.png)

![img](https://images2018.cnblogs.com/blog/1327126/201805/1327126-20180531202226978-1114842913.png)

注意：

1.numpy的ravel函数功能是将多维数组降为一维数组。参考博客：<https://blog.csdn.net/lanchunhui/article/details/50354978>

2.matplotlib.pyplot.hist函数主要是计算直方图。

hist函数原型：hist(x, bins=None, range=None, density=None, weights=None,  cumulative=False, bottom=None, histtype='bar', align='mid',  orientation='vertical', rwidth=None, log=False, color=None, label=None,  stacked=False, normed=None, hold=None, data=None, kwargs)

x参数表示是一个数组或一个序列，是指定每个bin(箱子)分布的数据

bins参数表示指定bin(箱子)的个数,也就是总共有几条条状图

range参数表示箱子的下限和上限。即横坐标显示的范围，范围之外的将被舍弃。

参考博客：<https://blog.csdn.net/u013571243/article/details/48998619>

3.enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。

4.cv2.calcHist的原型为：calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist

images参数表示输入图像，传入时应该用中括号[ ]括起来

channels参数表示传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，如果是彩色图像（有3个通道），那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[ ]传入。

mask参数表示掩膜图像。如果统计整幅图，那么为None。主要是如果要统计部分图的直方图，就得构造相应的掩膜来计算。

histSize参数表示灰度级的个数，需要中括号，比如[256]

ranges参数表示像素值的范围，通常[0,256]。此外，假如channels为[0,1],ranges为[0,256,0,180],则代表0通道范围是0-256,1通道范围0-180。

hist参数表示计算出来的直方图。

参考：<https://blog.csdn.net/YZXnuaa/article/details/79231817>

5.关于pyplot模块里plot()函数、xlim()函数等的用法参考：

<https://blog.csdn.net/cymy001/article/details/78344316>

<https://blog.csdn.net/chinwuforwork/article/details/51786967>

8.3、直方图的应用

代码如下：

```
#直方图的应用    直方图均衡化（即调整图像的对比度）   直方图即统计各像素点的频次
import cv2 as cv
#全局直方图均衡化
def eaualHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)    #opencv的直方图均衡化要基于单通道灰度图像
    cv.namedWindow('input_image', cv.WINDOW_NORMAL)
    cv.imshow('input_image', gray)
    dst = cv.equalizeHist(gray)                #自动调整图像对比度，把图像变得更清晰
    cv.namedWindow("eaualHist_demo", cv.WINDOW_NORMAL)
    cv.imshow("eaualHist_demo", dst)

#局部直方图均衡化
def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(5, (8,8))
    dst = clahe.apply(gray)
    cv.namedWindow("clahe_demo", cv.WINDOW_NORMAL)
    cv.imshow("clahe_demo", dst)

src = cv.imread('E:/imageload/rice.png')

eaualHist_demo(src)
clahe_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180601121025619-565628197.png)

注意：

1.cv2.equalizeHist函数原型：equalizeHist(src[, dst]) -> dst。函数equalizeHist的作用：直方图均衡化，提高图像质量。

2.直方图均衡化:如果一副图像的像素占有很多的灰度级而且分布均匀，那么这样的图像往往有高对比度和多变的灰度色调。直方图均衡化就是一种能仅靠输入图像直方图信息自动达到这种效果的变换函数。它的基本思想是对图像中像素个数多的灰度级进行展宽，而对图像中像素个数少的灰度进行压缩，从而扩展像元取值的动态范围，提高了对比度和灰度色调的变化，使图像更加清晰。

3.全局直方图均衡化可能得到是一种全局意义上的均衡化，但是有的时候这种操作并不是很好，会把某些不该调整的部分给调整了。Opencv中还有一种直方图均衡化，它是一种局部直方图均衡化，也就是是说把整个图像分成许多小块（比如按10*10作为一个小块），那么对每个小块进行均衡化。

4.createCLAHE函数原型：createCLAHE([, clipLimit[, tileGridSize]]) -> retval

clipLimit参数表示对比度的大小。

tileGridSize参数表示每次处理块的大小 。

5.

clahe = cv.createCLAHE(5, (8,8))       

dst = clahe.apply(gray)      #猜测：把clahe这种局部直方图均衡化应用到灰度图gray

8.4、直方图反向投影

代码如下：

```
#直方图反向投影技术（通过二维直方图反映，必须先把原图像转换为hsv）
import cv2 as cv

#计算H-S直方图
def back_projection_demo():
    sample = cv.imread("E:/imageload/sample.jpg")
    target = cv.imread("E:/imageload/target.jpg")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    cv.namedWindow("sample", cv.WINDOW_NORMAL)
    cv.imshow("sample", sample)
    cv.namedWindow("target", cv.WINDOW_NORMAL)
    cv.imshow("target", target)
    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 30], [0, 180, 0, 256])#计算样本直方图   [32, 30]越小，效果越好
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX) #规划到0-255之间
    dst = cv.calcBackProject([target_hsv], [0,1], roiHist, [0, 180, 0, 256], 1) #计算反向投影
    cv.namedWindow("back_projection_demo", cv.WINDOW_NORMAL)
    cv.imshow("back_projection_demo", dst)

back_projection_demo()
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180601154712217-1598657425.png)

注意：

1.归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。

归一化函数cv2.normalize原型：normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]) -> dst 

src参数表示输入数组。

dst参数表示输出与src相同大小的数组，支持原地运算。

alpha参数表示range normalization模式的最小值。

beta参数表示range normalization模式的最大值，不用于norm normalization(范数归一化)模式。

norm_type参数表示归一化的类型。

norm_type参数可以有以下的取值：

NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用。

NORM_INF:归一化数组的C-范数(绝对值的最大值)。

NORM_L1 ：归一化数组的L1-范数(绝对值的和)。

NORM_L2 ：归一化数组的(欧几里德)L2-范数。

参考博客：<https://blog.csdn.net/solomon1558/article/details/44689611>

2.反向投影用于在输入图像（通常较大）中查找特定图像（通常较小或者仅1个像素，以下将其称为模板图像）最匹配的点或者区域，也就是定位模板图像出现在输入图像的位置。

函数cv2.calcBackProject用来计算直方图反向投影。

函数原型：calcBackProject(images, channels, hist, ranges, scale[, dst]) -> dst

images参数表示输入图像（是HSV图像）。传入时应该用中括号[ ]括起来。

channels参数表示用于计算反向投影的通道列表，通道数必须与直方图维度相匹配。

hist参数表示输入的模板图像直方图。

ranges参数表示直方图中每个维度bin的取值范围 （即每个维度有多少个bin）。

scale参数表示可选输出反向投影的比例因子，一般取1。

参考博客：<https://blog.csdn.net/keith_bb/article/details/70154219>

九、模板匹配

​		百度百科：模板匹配是一种最原始、最基本的模式识别方法，研究某一特定对象物的图案位于图像的什么地方，进而识别对象物，这就是一个匹配问题。它是图像处理中最基本、最常用的匹配方法。模板匹配具有自身的局限性，主要表现在它只能进行平行移动，若原图像中的匹配目标发生旋转或大小变化，该算法无效。

​		简单来说，模板匹配就是在整个图像区域发现与给定子图像匹配的小块区域。

​		工作原理：在带检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度，匹配程度越大，两者相同的可能性越大。

代码如下：

```
#模板匹配
import cv2 as cv
import numpy as np
def template_demo():
    tpl =cv.imread("E:/imageload/sample1.jpg")
    target = cv.imread("E:/imageload/target1.jpg")
    cv.namedWindow('template image', cv.WINDOW_NORMAL)
    cv.imshow("template image", tpl)
    cv.namedWindow('target image', cv.WINDOW_NORMAL)
    cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]   #3种模板匹配方法
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)   #br是矩形右下角的点的坐标
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        cv.imshow("match-" + np.str(md), target)

template_demo()
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180601205028232-1613571876.png)

注意：

1.几种常见的模板匹配算法：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180601205720855-1784936563.png)

其中，

①TM_SQDIFF是平方差匹配；TM_SQDIFF_NORMED是标准平方差匹配。利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大。

②TM_CCORR是相关性匹配；TM_CCORR_NORMED是标准相关性匹配。采用模板和图像间的乘法操作,数越大表示匹配程度较高, 0表示最坏的匹配效果。

③TM_CCOEFF是相关性系数匹配；TM_CCOEFF_NORMED是标准相关性系数匹配。将模版对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,-1表示糟糕的匹配,0表示没有任何相关性(随机序列)。

总结：随着从简单的测量(平方差)到更复杂的测量(相关系数),我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价)。

参考：

<http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/histograms/template_matching/template_matching.html>

<https://blog.csdn.net/guduruyu/article/details/69231259>

补：个人认为参考的第一篇博客的关于模板匹配算法的原理有一点点点错误，模板图像应该是左上角开始，而不是从中心点开始。在左上角那个点开始计算匹配度，最后得出的最匹配的坐标点是模板图像左上角的位置（纯属个人觉得，如有错误，欢迎指出来）。

我认为模板匹配原理应该如下：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180602130109847-1130514291.png)

2.opencv的目标匹配函数为matchTemplate，函数原型为：matchTemplate(image, templ, method[, result[, mask]]) -> result

image参数表示待搜索源图像，必须是8位整数或32位浮点。

templ参数表示模板图像，必须不大于源图像并具有相同的数据类型。

method参数表示计算匹配程度的方法。

result参数表示匹配结果图像，必须是单通道32位浮点。如果image的尺寸为W x H，templ的尺寸为w x h，则result的尺寸为(W-w+1)x(H-h+1)。

3.opencv的函数minMaxLoc：在给定的矩阵中寻找最大和最小值，并给出它们的位置。 该功能不适用于多通道阵列。 如果您需要在所有通道中查找最小或最大元素，要先将阵列重新解释为单通道。

函数minMaxLoc原型为：minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc

src参数表示输入单通道图像。

mask参数表示用于选择子数组的可选掩码。

minVal参数表示返回的最小值，如果不需要，则使用NULL。

maxVal参数表示返回的最大值，如果不需要，则使用NULL。

minLoc参数表示返回的最小位置的指针（在2D情况下）； 如果不需要，则使用NULL。

maxLoc参数表示返回的最大位置的指针（在2D情况下）； 如果不需要，则使用NULL。

参考：<https://blog.csdn.net/liuqz2009/article/details/60869427>

 4.opencv的函数rectangle用于绘制矩形。函数原型为： rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img

img参数表示源图像。

pt1参数表示矩形的一个顶点。

pt2参数表示与pt1相对的对角线上的另一个顶点 。

color参数表示矩形线条颜色 (RGB) 或亮度（灰度图像 ）。

thickness参数表示组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形。

lineType参数表示线条的类型。

shift参数表示坐标点的小数点位数。

十、图像二值化

简介：图像二值化就是将图像上的[像素](https://baike.baidu.com/item/像素)点的[灰度值](https://baike.baidu.com/item/灰度值)设置为0或255，也就是将整个图像呈现出明显的黑白效果的过程。

普通图像二值化

代码如下：

```
import cv2 as cv
import numpy as np

#全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    cv.imshow("binary0", binary)

#局部阈值
def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
    cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    cv.imshow("binary1", binary)

#用户自己计算阈值
def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    cv.imshow("binary2", binary)

src = cv.imread('E:/imageload/kobe.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
threshold_demo(src)
local_threshold(src)
custom_threshold(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180602182006421-500970779.png)

注意：

1.全局阈值

①OpenC的threshold函数进行全局阈值。其函数原型为：threshold(src, thresh, maxval, type[, dst]) -> retval, dst

src参数表示输入图像（多通道，8位或32位浮点）。

thresh参数表示阈值。

maxval参数表示与THRESH_BINARY和THRESH_BINARY_INV阈值类型一起使用设置的最大值。

type参数表示阈值类型。

retval参数表示返回的阈值。若是全局固定阈值算法，则返回thresh参数值。若是全局自适应阈值算法，则返回自适应计算得出的合适阈值。

dst参数表示输出与src相同大小和类型以及相同通道数的图像。

②type参数阈值类型这部分参考博客：<https://blog.csdn.net/iracer/article/details/49232703>  ，写的很不错。

阈值类型：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180602184734982-2078253219.png)

阈值类型图示：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180602184900884-1601429071.png)

③type参数单独选择上述五种阈值类型时，是固定阈值算法，效果比较差。

此外还有自适应阈值算法：（自适应计算合适的阈值，而不是固定阈值）

比如结合cv.THRESH_OTSU，写成cv.THRESH_BINARY | cv.THRESH_OTSU。例子：ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) #大律法,全局自适应阈值,第二个参数值0可改为任意数字但不起作用。 

比如结合cv.THRESH_TRIANGLE，写成cv.THRESH_BINARY | cv.THRESH_TRIANGLE。例子：ret,  binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY |  cv.THRESH_TRIANGLE) #TRIANGLE法,全局自适应阈值,第二个参数值0可改为任意数字但不起作用，适用于单个波峰。  

补：

cv.THRESH_OTSU和cv.THRESH_TRIANGLE也可单独使用，不一定要写成和固定阈值算法结合的形式。单独写和结合起来写，都是自适应阈值算法优先。

例子：ret, binary = cv.threshold(gray, 0, 255,  cv.THRESH_OTSU) #大律法       ret, binary = cv.threshold(gray, 0, 255,  cv.THRESH_TRIANGLE) #TRIANGLE法  

2.局部阈值

OpenCV的adaptiveThreshold函数进行局部阈值。函数原型为：adaptiveThreshold(src,  maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst

src参数表示输入图像（8位单通道图像）。

maxValue参数表示使用 THRESH_BINARY 和 THRESH_BINARY_INV 的最大值.

adaptiveMethod参数表示自适应阈值算法，平均 （ADAPTIVE_THRESH_MEAN_C）或高斯（ADAPTIVE_THRESH_GAUSSIAN_C）。

thresholdType参数表示阈值类型，必须为THRESH_BINARY或THRESH_BINARY_INV的阈值类型。

blockSize参数表示块大小（奇数且大于1，比如3，5，7........ ）。

C参数是常数，表示从平均值或加权平均值中减去的数。 通常情况下，这是正值，但也可能为零或负值。

补：在使用平均和高斯两种算法情况下，通过计算每个像素周围blockSize x  blockSize大小像素块的加权均值并减去常量C即可得到自适应阈值。如果使用平均的方法，则所有像素周围的权值相同；如果使用高斯的方法，则每个像素周围像素的权值则根据其到中心点的距离通过高斯方程得到。

参考：<https://blog.csdn.net/guduruyu/article/details/68059450>

3.numpy的reshape函数是给数组一个新的形状而不改变其数据，函数原型：reshape(a, newshape, order='C')

a参数表示需要重新形成的原始数组。

newshape参数表示int或int类型元组（tuple），若为(1, 3),表示生成的新数组是1行3列。

order参数表表示使用此索引顺序读取a的元素，并使用此索引顺序将元素放置到重新形成的数组中。

函数返回值：如果可能的话，这将是一个新的视图对象; 否则，它会成为副本。

十一、图像金字塔

简介：图像金字塔是图像中多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构。简单来说，图像金字塔就是用来进行图像缩放的。

进行图像缩放可以用图像金字塔，也可以使用resize函数进行缩放，后者效果更好。这里只是对图像金字塔做一些简单了解。

两种类型的金字塔：

①高斯金字塔：用于下采样。高斯金字塔是最基本的图像塔。原理：首先将原图像作为最底层图像G0（高斯金字塔的第0层），利用高斯核（5*5）对其进行卷积，然后对卷积后的图像进行下采样（去除偶数行和列）得到上一层图像G1，将此图像作为输入，重复卷积和下采样操作得到更上一层图像，反复迭代多次，形成一个金字塔形的图像数据结构，即高斯金字塔。

②拉普拉斯金字塔：用于重建图像，也就是预测残差，对图像进行最大程度的还原。比如一幅小图像重建为一幅大图，原理：用高斯金字塔的每一层图像减去其上一层图像上采样并高斯卷积之后的预测图像，得到一系列的差值图像即为 LP 分解图像。

两种类型的采样：

①上采样：就是图片放大（所谓上嘛，就是变大），使用PryUp函数。    上采样步骤：先将图像在每个方向放大为原来的两倍，新增的行和列用0填充，再使用先前同样的内核与放大后的图像卷积，获得新增像素的近似值。     

②下采样：就是图片缩小（所谓下嘛，就是变小），使用PryDown函数。下采样将步骤：先对图像进行高斯内核卷积 ，再将所有偶数行和列去除。

总之，上、下采样都存在一个严重的问题，那就是图像变模糊了，因为缩放的过程中发生了信息丢失的问题。要解决这个问题，就得用拉普拉斯金字塔。

参考博客：

<https://www.cnblogs.com/skyfsm/p/6876732.html>

<https://blog.csdn.net/app_12062011/article/details/52471299>

代码如下：

```
import cv2 as cv
#高斯金字塔
def pyramid_demo(image):
    level = 3      #设置金字塔的层数为3
    temp = image.copy()  #拷贝图像
    pyramid_images = []  #建立一个空列表
    for i in range(level):
        dst = cv.pyrDown(temp)   #先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        pyramid_images.append(dst)  #在列表末尾添加新的对象
        cv.imshow("pyramid"+str(i), dst)
        temp = dst.copy()
    return pyramid_images
#拉普拉斯金字塔
def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)    #做拉普拉斯金字塔必须用到高斯金字塔的结果
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize = image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize = pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
src = cv.imread('E:/imageload/zixia.jpg')
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
lapalian_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180603133200199-1430278762.png)

注意：

1.opencv的pyrDown函数先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）。其函数原型为：pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst

src参数表示输入图像。

dst参数表示输出图像，它与src类型、大小相同。

dstsize参数表示降采样之后的目标图像的大小。它是有默认值的，如果我们调用函数的时候不指定第三个参数，那么这个值是按照 Size((src.cols+1)/2, (src.rows+1)/2) 计算的。而且不管你自己如何指定这个参数，一定必须保证满足以下关系式：|dstsize.width * 2 - src.cols| ≤ 2;  |dstsize.height * 2 - src.rows| ≤ 2。也就是说降采样的意思其实是把图像的尺寸缩减一半，行和列同时缩减一半。

borderType参数表示表示图像边界的处理方式。

2.opencv的pyrUp函数先对图像进行升采样（将图像尺寸行和列方向增大一倍），然后再进行高斯平滑。其函数原型为：pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst

src参数表示输入图像。

dst参数表示输出图像，它与src类型、大小相同。

dstsize参数表示降采样之后的目标图像的大小。在默认的情况下，这个尺寸大小是按照 Size(src.cols*2, (src.rows*2) 来计算的。如果你自己要指定大小，那么一定要满足下面的条件：

|dstsize.width - src.cols * 2| ≤ (dstsize.width mod 2);  //如果width是偶数，那么必须dstsize.width是src.cols的2倍

|dstsize.height - src.rows * 2| ≤ (dstsize.height mod 2);

borderType参数表示表示图像边界的处理方式。

参考：

<https://blog.csdn.net/woainishifu/article/details/62888228>

<https://blog.csdn.net/poem_qianmo/article/details/26157633>

 

注意：拉普拉斯金字塔时，图像大小必须是2的n次方\*2的n次方，不然会报错？（只要图像长和宽相等即可，并不非要是2的n次方\*2的n次方，至少我这么做没问题，也不知道为什么都说图像大小必须是2的n次方\*2的n次方，求知道的大佬告知一波！）

十二、图像梯度

简介：图像梯度可以把图像看成二维离散函数，图像梯度其实就是这个二维离散函数的求导。

Sobel算子是普通一阶差分，是基于寻找梯度强度。拉普拉斯算子（二阶差分）是基于过零点检测。通过计算梯度，设置阀值，得到边缘图像。

以下各种算子的原理可参考：<https://blog.csdn.net/poem_qianmo/article/details/25560901>

12.1、Sobel算子

代码如下：

```
import cv2 as cv
#Sobel算子
def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  #x方向上的梯度
    cv.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    cv.imshow("gradient", gradxy)

src = cv.imread('E:/imageload/liu.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
sobel_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180603200054758-1613821748.png)

 注意：

1.Sobel算子用来计算图像灰度函数的近似梯度。Sobel算子根据像素点上下、左右邻点灰度加权差，在边缘处达到极值这一现象检测边缘。对噪声具有平滑作用，提供较为精确的边缘方向信息，边缘定位精度不够高。当对精度要求不是很高时，是一种较为常用的边缘检测方法。

2.Sobel具有平滑和微分的功效。即：Sobel算子先将图像横向或纵向平滑，然后再纵向或横向差分，得到的结果是平滑后的差分结果。

OpenCV的Sobel函数原型为：Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

src参数表示输入需要处理的图像。

ddepth参数表示输出图像深度，针对不同的输入图像，输出目标图像有不同的深度。

　　具体组合如下： 
　　src.depth() = CV_8U, 取ddepth =-1/CV_16S/CV_32F/CV_64F （一般源图像都为CV_8U，为了避免溢出，一般ddepth参数选择CV_32F）
　　src.depth() = CV_16U/CV_16S, 取ddepth =-1/CV_32F/CV_64F 
　　src.depth() = CV_32F, 取ddepth =-1/CV_32F/CV_64F 
　　src.depth() = CV_64F, 取ddepth = -1/CV_64F 
　　注：ddepth =-1时，代表输出图像与输入图像相同的深度。 

dx参数表示x方向上的差分阶数，1或0 。

dy参数表示y 方向上的差分阶数，1或0 。

dst参数表示输出与src相同大小和相同通道数的图像。

ksize参数表示Sobel算子的大小，必须为1、3、5、7。

scale参数表示缩放导数的比例常数，默认情况下没有伸缩系数。

delta参数表示一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中。

borderType表示判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。

参考：

<https://blog.csdn.net/streamchuanxi/article/details/51542141>

<https://blog.csdn.net/sunny2038/article/details/9170013>

Sobel算子原理：<https://www.cnblogs.com/lancidie/archive/2011/07/17/2108885.html>

2.OpenCV的convertScaleAbs函数使用线性变换转换输入数组元素成8位无符号整型。函数原型：convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst

src参数表示原数组。

dst参数表示输出数组 (深度为 8u)。

alpha参数表示比例因子。

beta参数表示原数组元素按比例缩放后添加的值。

3.OpenCV的addWeighted函数是计算两个数组的加权和。函数原型：addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst

src1参数表示需要加权的第一个输入数组。

alpha参数表示第一个数组的权重。

src2参数表示第二个输入数组，它和第一个数组拥有相同的尺寸和通道数。

 beta参数表示第二个数组的权重。

gamma参数表示一个加到权重总和上的标量值。

dst参数表示输出的数组，它和输入的两个数组拥有相同的尺寸和通道数。

dtype参数表示输出数组的可选深度。当两个输入数组具有相同的深度时，这个参数设置为-1（默认值），即等同于src1.depth（）。

 

12.2、Scharr算子

代码如下：

```
import cv2 as cv
#Scharr算子(Sobel算子的增强版，效果更突出)
def Scharr_demo(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  #x方向上的梯度
    cv.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)
src = cv.imread('E:/imageload/liu.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
Scharr_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180603210924551-515053573.png)

注意：

Scharr算子也是计算x或y方向上的图像差分。OpenCV的Scharr函数原型为：Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst

参数和Sobel算子的几乎差不多，意思也一样，只是没有ksize大小。

Scharr原理参考：[https://www.tony4ai.com/DIP-6-6-%E7%81%B0%E5%BA%A6%E5%9B%BE%E5%83%8F-%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2-Scharr%E7%AE%97%E5%AD%90/](https://www.tony4ai.com/DIP-6-6-灰度图像-图像分割-Scharr算子/)

 

12.3、拉普拉斯算子

代码如下：

```
import cv2 as cv
#拉普拉斯算子
def Laplace_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("Laplace_demo", lpls)
src = cv.imread('E:/imageload/liu.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
Laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180604092423301-832975068.png)

注意：

1.拉普拉斯算子（Laplace Operator）是n维[欧几里德空间](https://baike.baidu.com/item/欧几里德空间)中的一个二阶微分算子，定义为[梯度](https://baike.baidu.com/item/梯度)（▽f）的[散度](https://baike.baidu.com/item/散度)（▽·f）。

2.OpenCV的Laplacian函数原型为：Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

src参数表示输入需要处理的图像。

ddepth参数表示输出图像深度，针对不同的输入图像，输出目标图像有不同的深度。

　　具体组合如下： 
　　src.depth() = CV_8U, 取ddepth =-1/CV_16S/CV_32F/CV_64F （一般源图像都为CV_8U，为了避免溢出，一般ddepth参数选择CV_32F）
　　src.depth() = CV_16U/CV_16S, 取ddepth =-1/CV_32F/CV_64F 
　　src.depth() = CV_32F, 取ddepth =-1/CV_32F/CV_64F 
　　src.depth() = CV_64F, 取ddepth = -1/CV_64F 
　　注：ddepth =-1时，代表输出图像与输入图像相同的深度。 

dst参数表示输出与src相同大小和相同通道数的图像。

ksize参数表示用于计算二阶导数滤波器的孔径大小，大小必须是正数和奇数。

scale参数表示计算拉普拉斯算子值的比例因子，默认情况下没有伸缩系数。

delta参数表示一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中。

borderType表示判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。

补：

这里ksize参数默认值为1，此时Laplacian()函数采用以下3x3的孔径：

​                                             ![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180604094323326-1207656753.png)

十三、Canny边缘检测

简介：

1.Canny[边缘检测](https://baike.baidu.com/item/边缘检测)算子是John F. Canny于 1986 年开发出来的一个多级边缘检测算法。

2.Canny 的目标是找到一个最优的[边缘检测](https://baike.baidu.com/item/边缘检测)算法，最优边缘检测的含义是：

好的检测- 算法能够尽可能多地标识出图像中的实际边缘。

好的定位- 标识出的边缘要尽可能与实际图像中的实际边缘尽可能接近。

最小响应- 图像中的边缘只能标识一次，并且可能存在的图像噪声不应标识为边缘。

3.算法步骤：

　　①高斯模糊 - GaussianBlur
　　②灰度转换 - cvtColor
　　③计算梯度 – Sobel/Scharr
　　④非最大信号抑制
　　⑤高低阈值输出二值图像

代码如下：

```
 1 #Canny边缘提取
 2 import cv2 as cv
 3 def edge_demo(image):
 4     blurred = cv.GaussianBlur(image, (3, 3), 0)
 5     gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
 6     # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
 7     # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
 8     # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
 9     edge_output = cv.Canny(gray, 50, 150)
10     cv.imshow("Canny Edge", edge_output)
11     dst = cv.bitwise_and(image, image, mask= edge_output)
12     cv.imshow("Color Edge", dst)
13 src = cv.imread('E:/imageload/liu.jpg')
14 cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
15 cv.imshow('input_image', src)
16 edge_demo(src)
17 cv.waitKey(0)
18 cv.destroyAllWindows()
```

注：其中第9行代码可以用6、7、8行代码代替！两种方法效果一样。

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180604145453045-302369180.png)

 注意：

OpenCV的Canny函数用于在图像中查找边缘，其函数原型有两种：

①直接调用Canny算法在单通道灰度图像中查找边缘，

其函数原型为：Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

image参数表示8位输入图像。

threshold1参数表示设置的低阈值。

threshold2参数表示设置的高阈值，一般设定为低阈值的3倍 (根据Canny算法的推荐)。

edges参数表示输出边缘图像，单通道8位图像。

apertureSize参数表示Sobel算子的大小。

L2gradient参数表示一个布尔值，如果为真，则使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开方），否则使用L1范数（直接将两个方向导数的绝对值相加）。

②使用带自定义图像渐变的Canny算法在图像中查找边缘，

其函数原型为：Canny(dx, dy, threshold1, threshold2[, edges[, L2gradient]]) -> edges

dx参数表示输入图像的x导数（x导数满足16位，选择CV_16SC1或CV_16SC3）

dy参数表示输入图像的y导数（y导数满足16位，选择CV_16SC1或CV_16SC3）。

threshold1参数表示设置的低阈值。

threshold2参数表示设置的高阈值，一般设定为低阈值的3倍 (根据Canny算法的推荐)。

edges参数表示输出边缘图像，单通道8位图像。

L2gradient参数表示L2gradient参数表示一个布尔值，如果为真，则使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开方），否则使用L1范数（直接将两个方向导数的绝对值相加）。

参考：

Canny算子原理：

https://www.cnblogs.com/techyan1990/p/7291771.html

http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

https://blog.csdn.net/sunny2038/article/details/9202641

十四、直线检测

简介：

1.霍夫变换(Hough Transform) 霍夫变换是图像处理中从图像中识别几何形状的基本方法之一，应用很广泛，也有很多改进算法。主要用来从图像中分离出具有某种相同特征的几何形状（如，直线，圆等）。最基本的霍夫变换是从黑白图像中检测直线([线段](https://baike.baidu.com/item/线段))。

2.Hough变换的原理是将特定图形上的点变换到一组参数空间上，根据参数空间点的累计结果找到一个极大值对应的解，那么这个解就对应着要寻找的几何形状的参数（比如说直线，那么就会得到直线的斜率k与常熟b，圆就会得到圆心与半径等等）

3.霍夫线变换是一种用来寻找直线的方法。用霍夫线变换之前, 首先需要对图像进行边缘检测的处理，也即霍夫线变换的直接输入只能是边缘二值图像。

4.霍夫直线检测的具体原理参见：

<https://blog.csdn.net/ycj9090900/article/details/52944708>

<http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html>

<http://lib.csdn.net/article/opencv/24201>

代码如下：

```
#直线检测
#使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成
import cv2 as cv
import numpy as np

#标准霍夫线变换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    cv.imshow("image-lines", image)

#统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo",image)

src = cv.imread('E:/imageload/louti.jpg')
print(src.shape)
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) 
cv.imshow('input_image', src)
line_detection(src)
src = cv.imread('E:/imageload/louti.jpg') #调用上一个函数后，会把传入的src数组改变，所以调用下一个函数时，要重新读取图片
line_detect_possible_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201806/1327126-20180609202921824-770088810.png)

注意：

1.opencv的HoughLines函数是标准霍夫线变换函数，该函数的功能是通过一组参数对 ![(\theta, r_{\theta})](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/7e065ce45efb160a92773b6c4aad65a309d7535e.png) 的集合来表示检测到的直线，其函数原型为：HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines

image参数表示边缘检测的输出图像，该图像为单通道8位二进制图像。

rho参数表示参数极径 ![r](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/b55ca7a0aa88ab7d58f4fc035317fdac39b17861.png) 以像素值为单位的分辨率，这里一般使用1像素。

theta参数表示参数极角 ![\theta](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/52e8ed7a3ba22130ad3984eb2cd413406475a689.png) 以弧度为单位的分辨率，这里使用1度。

threshold参数表示检测一条直线所需最少的曲线交点。

lines参数表示储存着检测到的直线的参数对 ![(r,\theta)](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/f07080129914ac008f0eb45ed5f7efa28bb1e7c6.png) 的容器 。

srn参数、stn参数默认都为0。如果srn = 0且stn = 0，则使用经典的Hough变换。

min_theta参数表示对于标准和多尺度Hough变换，检查线条的最小角度。

max_theta参数表示对于标准和多尺度Hough变换，检查线条的最大角度。

2.opencv的HoughLinesP函数是统计概率霍夫线变换函数，该函数能输出检测到的直线的端点 ![(x_{0}, y_{0}, x_{1}, y_{1})](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/20eb30037d99342a5887bea2a086e1b39e78904a.png)，其函数原型为：HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines

image参数表示边缘检测的输出图像，该图像为单通道8位二进制图像。

rho参数表示参数极径 ![r](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/b55ca7a0aa88ab7d58f4fc035317fdac39b17861.png) 以像素值为单位的分辨率，这里一般使用 1 像素。

theta参数表示参数极角 ![\theta](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/52e8ed7a3ba22130ad3984eb2cd413406475a689.png) 以弧度为单位的分辨率，这里使用 1度。

threshold参数表示检测一条直线所需最少的曲线交点。

lines参数表示储存着检测到的直线的参数对 ![(x_{start}, y_{start}, x_{end}, y_{end})](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/baee84de274f174bbcc7d67e86f0dea7b49a0af9.png) 的容器，也就是线段两个端点的坐标。

minLineLength参数表示能组成一条直线的最少点的数量，点数量不足的直线将被抛弃。

maxLineGap参数表示能被认为在一条直线上的亮点的最大距离。

十五、圆检测

简介：

1.霍夫圆变换的基本原理和霍夫线变换原理类似，只是点对应的二维极径、极角空间被三维的圆心和半径空间取代。在标准霍夫圆变换中，原图像的边缘图像的任意点对应的经过这个点的所有可能圆在三维空间用圆心和半径这三个参数来表示，其对应一条三维空间的曲线。对于多个边缘点，点越多，这些点对应的三维空间曲线交于一点的数量越多，那么他们经过的共同圆上的点就越多，类似的我们也就可以用同样的阈值的方法来判断一个圆是否被检测到，这就是标准霍夫圆变换的原理，  但也正是在三维空间的计算量大大增加的原因，标准霍夫圆变化很难被应用到实际中。

2.OpenCV实现的是一个比标准霍夫圆变换更为灵活的检测方法——霍夫梯度法，该方法运算量相对于标准霍夫圆变换大大减少。其检测原理是依据圆心一定是在圆上的每个点的模向量上，这些圆上点模向量的交点就是圆心，霍夫梯度法的第一步就是找到这些圆心，这样三维的累加平面就又转化为二维累加平面。第二步是根据所有候选中心的边缘非0像素对其的支持程度来确定半径。注：模向量即是圆上点的切线的垂直线。

![img](https://images2018.cnblogs.com/blog/1327126/201807/1327126-20180729195340696-858749488.png)

霍夫圆检测原理参考：

<http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html>

<https://blog.csdn.net/dcrmg/article/details/52506538>

代码如下：

```
#霍夫圆检测
import cv2 as cv
import numpy as np

def detect_circles_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)   #边缘保留滤波EPF
    cimage = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles)) #把circles包含的圆心和半径的值变成整数
    for i in circles[0, : ]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  #画圆
        cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  #画圆心
    cv.imshow("circles", image)

src = cv.imread('E:/imageload/coins.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
detect_circles_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201807/1327126-20180729195752449-1817988833.png)

 注意：

1.OpenCV的霍夫圆变换函数原型为：HoughCircles(image, method, dp, minDist[,  circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles

image参数表示8位单通道灰度输入图像矩阵。

method参数表示圆检测方法，目前唯一实现的方法是HOUGH_GRADIENT。

dp参数表示累加器与原始图像相比的分辨率的反比参数。例如，如果dp = 1，则累加器具有与输入图像相同的分辨率。如果dp=2，累加器分辨率是元素图像的一半，宽度和高度也缩减为原来的一半。

minDist参数表示检测到的两个圆心之间的最小距离。如果参数太小，除了真实的一个圆圈之外，可能错误地检测到多个相邻的圆圈。如果太大，可能会遗漏一些圆圈。

circles参数表示检测到的圆的输出向量，向量内第一个元素是圆的横坐标，第二个是纵坐标，第三个是半径大小。

param1参数表示Canny边缘检测的高阈值，低阈值会被自动置为高阈值的一半。

param2参数表示圆心检测的累加阈值，参数值越小，可以检测越多的假圆圈，但返回的是与较大累加器值对应的圆圈。

minRadius参数表示检测到的圆的最小半径。

maxRadius参数表示检测到的圆的最大半径。

2.OpenCV画圆的circle函数原型：circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img

img参数表示源图像。

center参数表示圆心坐标。

radius参数表示圆的半径。

color参数表示设定圆的颜色。

thickness参数：如果是正数，表示圆轮廓的粗细程度。如果是负数，表示要绘制实心圆。

lineType参数表示圆线条的类型。

shift参数表示圆心坐标和半径值中的小数位数。

十六、轮廓发现

简介：轮廓发现是基于图像边缘提取的基础寻找对象轮廓的方法，所以边缘提取的阈值选定会影响最终轮廓发现结果。

代码如下：

```
import cv2 as cv
import numpy as np
def contours_demo(image):
    dst = cv.GaussianBlur(image, (3, 3), 0) #高斯模糊去噪
    gray = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) #用大律法、全局自适应阈值方法进行图像二值化
    cv.imshow("binary image", binary)
    cloneTmage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        print(i)
    cv.imshow("contours", image)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), -1)
    cv.imshow("pcontours", image)
src = cv.imread('E:/imageload/coins.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
contours_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

运行结果：

![img](https://images2018.cnblogs.com/blog/1327126/201808/1327126-20180812102112514-1647676425.png)

注意：

1.Opencv发现轮廓的函数原型为：findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy

image参数表示8位单通道图像矩阵，可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像。

mode参数表示轮廓检索模式：

①CV_RETR_EXTERNAL：只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略。

②CV_RETR_LIST：检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓。

③CV_RETR_CCOMP：检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层。

④CV_RETR_TREE：检测所有轮廓，所有轮廓建立一个等级树结构，外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。

method参数表示轮廓的近似方法：

①CV_CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1。

②CV_CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息。

③CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法。

contours参数是一个list，表示存储的每个轮廓的点集合。

hierarchy参数是一个list,list中元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0]  ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。

offset参数表示每个轮廓点移动的可选偏移量。

 2.Opencv绘制轮廓的函数原型为：drawContours(image, contours, contourIdx, color[,  thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image

imgae参数表示目标图像。

contours参数表示所有输入轮廓。

contourIdx参数表示绘制轮廓list中的哪条轮廓， 如果是负数，则绘制所有轮廓。

color参数表示轮廓的颜色。

thickness参数表示绘制的轮廓线条粗细，如果是负数，则绘制轮廓内部。

lineType参数表示线型。

hierarchy参数表示有关层次结构的可选信息。

maxLevel参数表示绘制轮廓的最大级别。 如果为0，则仅绘制指定的轮廓。 如果为1，则该函数绘制轮廓和所有嵌套轮廓。 如果为2，则该函数绘制轮廓，所有嵌套轮廓，所有嵌套到嵌套的轮廓，等等。 仅当有可用的层次结构时才考虑此参数。

offset参数表示可选的轮廓偏移参数，该参数可按指定的方式移动所有绘制的轮廓。







​		以上整理自：

https://www.cnblogs.com/FHC1994/p/8993237.html