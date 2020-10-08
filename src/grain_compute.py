import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from IPython import embed

#directories= ['C', 'N', 'Y']
directories= ['C', 'Y']

def imagecrop(close_result,box):
    xs = [x[1] for x in box]
    ys = [x[0] for x in box]
                #print(xs)
                #print(min(xs),max(xs),min(ys),max(ys))
    cropimage = close_result[min(xs):max(xs),min(ys):max(ys)]
                #print(cropimage.shape)
                #cv2.imwrite('cropimage.png',cropimage)
    # embed()
    return cropimage

# find the appropriate contour
# TODO@branzhu@yitianjiang, consider wether the largest contour is out of the image
def findProperCounter(contours, img = None):
    max_ = 0
    max_contour = None
    for contour in contours:
        #embed()
        if contour.shape[0] > max_:
            max_ = contour.shape[0]
            max_contour = contour

    return max_contour

for directory in directories:
    new_file_name = os.path.join('cut'+directory, 'xy.csv')
    OUT = open(new_file_name, 'w')
    for filename in os.listdir(directory):
        if filename.endswith(".bmp"):
            print(filename)
            gray_img = cv.imread(os.path.join(directory, filename), cv.IMREAD_GRAYSCALE)
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
            #print(median)
            ret,thresh = cv.threshold(median_img,120,255,cv.THRESH_BINARY) #The method returns two outputs. The first is the threshold that was used and the second output is the thresholded image.
            #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #print(os.path.join(directory, filename))
            #for cont in contours:
            #cv.drawContours(thresh, contours =contours, contourIdx=1, color=128, thickness=-1)
            kernel = np.ones((5,5), np.uint8)#设置开闭运算卷积核
            open_result = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            close_result = cv.morphologyEx(open_result, cv.MORPH_CLOSE, kernel)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contour = findProperCounter(contours)
            OUT.write(filename + ',')
            area = cv.contourArea(contour)
            OUT.write(str(area))
            OUT.write(',')
            #embed()
            perimeter = cv.arcLength(contour,True)
            OUT.write(str(perimeter))
            OUT.write(',')
            circularity=4*3.141593*area/pow(perimeter,2)
            OUT.write(str(circularity))
            OUT.write(',')
            compactness=pow(perimeter,2)/area
            OUT.write(str(compactness))
            OUT.write(',')

            rect = cv.minAreaRect(contour)# 找面积最小的矩形
            box = cv.boxPoints(rect) # 得到最小矩形的坐标
            box = np.int0(box)# 标准化坐标到整数
            # cv.drawContours(close_result, [box], 0, (0, 0, 255), 3) # 画出边界

            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            # the perspective transformation matrix
            M = cv.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv.warpPerspective(close_result, M, (width, height))
            if(warped.shape[0] > warped.shape[1]):
                length = warped.shape[0]
                width = warped.shape[1]
            else:
                length =  warped.shape[1]
                width = warped.shape[0]
            lwratio =  width/length
            OUT.write(str(lwratio))
            OUT.write(',')

            merarea = length*width
            rectness = area/merarea
            OUT.write(str(rectness))
            OUT.write(',')
            #print(warped.shape)
            OUT.write("{},{}\n".format(length, width))
            
            
            #embed()
            #cut_img = close_result[y:y+h, x:x+w]
            
            #cv.imwrite(new_file_name,thresh)
            # break
            #print(new_file_name)
       # else:
           # continue
    




