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
            kernel = np.ones((5,5), np.uint8)
            open_result = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            close_result = cv.morphologyEx(open_result, cv.MORPH_CLOSE, kernel)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contour = findProperCounter(contours)
            if False:
                gaussian = cv.GaussianBlur(gray_img, (3,3), 0)
                Canny = cv.Canny(gaussian, 50, 150) 

                max_id, max_counter = 0,  -1
                for idx,contour in enumerate(contours):
                    try:
                        perimeter = cv.contourArea(Canny)
                    except:
                        print('except')
                        perimeter = 0
                    if perimeter > max_counter:
                        max_counter = perimeter
                        max_id = idx
            #embed()
            #x, y, w, h = cv.boundingRect(contour) 
            #print(x, y, w, h)
            # cv.rectangle(close_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect) 
            box = np.int0(box)
            # cv.drawContours(close_result, [box], 0, (0, 0, 255), 3) 

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
            #print(warped.shape)
            
            
            #embed()
            #cut_img = close_result[y:y+h, x:x+w]
            new_file_name = os.path.join('cut'+directory, filename)
            print(new_file_name)
            #plt.show(cut_img)
            #embed()
            cv.imwrite(new_file_name,warped)
            
            #cv.imwrite(new_file_name,thresh)
            # break
            #print(new_file_name)
       # else:
           # continue
    




