import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
from IPython import embed
import imutils
gray_img = cv.imread('./cutY/Y001.bmp', cv.IMREAD_GRAYSCALE)
#Roberts算子
#kernelx = np.array([[-1,0],[0,1]], dtype=int)
#kernely = np.array([[0,-1],[1,0]], dtype=int)
#x = cv.filter2D(gray_img, cv.CV_16S, kernelx)
#y = cv.filter2D(gray_img, cv.CV_16S, kernely)
#转uint8 
#absX = cv.convertScaleAbs(x)      
#absY = cv.convertScaleAbs(y)    
#Roberts = cv.addWeighted(absX,0.5,absY,0.5,0)
#Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
#Prewitt = cv.addWeighted(absX,0.5,absY,0.5,0)
gaussian = cv.GaussianBlur(gray_img, (3,3), 0)
 
#dst = cv.Laplacian(gaussian, cv.CV_16S, ksize = 3)
#LOG = cv.convertScaleAbs(dst)
Canny = cv.Canny(gaussian, 50, 150) 
#plt.rcParams['font.sans-serif']=['SimHei']
a,b = cv.findContours(Canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
area = cv.contourArea(a[0])
print(area)

#embed()
perimeter = cv.arcLength(a[0],True)
print(perimeter)
circularity=4*3.141593*area/pow(perimeter,2)
print(circularity)
compactness=pow(perimeter,2)/area
print(compactness)
#显示图形
#titles = [u'原始图像', u'Canny算子']  
#images = [gray_img, Canny]  
#for i in range(2):  
#   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
#   plt.title(titles[i])  
#   plt.xticks([]),plt.yticks([])  
#plt.show()
#embed()
