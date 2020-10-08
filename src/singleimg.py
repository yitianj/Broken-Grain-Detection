import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from IPython import embed

def imagecrop(image,box):
      xs = [x[1] for x in box]
      ys = [x[0] for x in box]
      print(xs)
      print(min(xs),max(xs),min(ys),max(ys))
      cropimage = close_result[min(xs):max(xs),min(ys):max(ys)]
      print(cropimage.shape)
      #cv2.imwrite('cropimage.png',cropimage)
      return cropimage
      
#src = cv2.imread('Y001.bmp')
gray_img = cv.imread('Y003.bmp', cv.IMREAD_GRAYSCALE)
height = gray_img.shape[0]
width = gray_img.shape[1]
for i in range(height):
    for j in range(width):
        
        if (int(gray_img[i,j]*1.5) > 255):
            gray = 255
        else:
            gray = int(gray_img[i,j]*1.5)
            
gray_img[i,j] = np.uint8(gray)
median_img = cv.medianBlur(gray_img,9)
ret,thresh = cv.threshold(median_img,120,255,cv.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
open_result = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
close_result = cv.morphologyEx(open_result, cv.MORPH_CLOSE, kernel)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv.boundingRect(contours[1]) 
#cv.rectangle(close_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
rect = cv.minAreaRect(contours[1])
box = cv.boxPoints(rect) 
box = np.int0(box)


cut_img = imagecrop(close_result,box)
cv.imwrite("contour_cut_img.bmp",cut_img)
#cv2.imshow("src",src)
#cv2.waitkey(0)
#cv2.destroyAllWindows()
#plt.hist(thresh.ravel(),256)
#plt.show()
#embed()
