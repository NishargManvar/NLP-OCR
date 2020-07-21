#import necessary files
import cv2
import nltk
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import glob
from skimage.io import imread

#Initialisations
area=0
max=area

#Opening File for editing
img = imread('/content/document_basic.png',as_gray=False) #change image adress as needed.
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Applying canny to find edges
edges = cv2.Canny(img,300,350)

#Dilating with a big kernel in an attempt to have proper boundaries
kernel = np.ones((8,8),np.uint8)
opening = cv2.dilate(edges,kernel,iterations = 1)

#Finding contours and finding the contour with the biggest area
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    rect = cv2.minAreaRect(c)
    w = rect[1][0]
    h = rect[1][1]
    area = w*h
    if area > max:
      max_c = c
      max = area
      x_max = rect[0][0]
      y_max = rect[0][1]
      w_max = w
      h_max = h
      rect_max = rect

box = cv2.boxPoints(rect_max) #box has the co-ordinates of the rectsngle to be warped
box = np.int0(box)
box_max = np.array([box[1],box[2],box[3],box[0]],dtype = "float32") #co-ordinates in clockwise order
opening = cv2.drawContours(opening,[box],0,(255,255,255),2) #draw contour which is chossen for warping on the image 

maxWidth = w_max
maxHeight = h_max

#dst has destination co-ordinates where contour is to be warped
dst = np.array([[0,0],[maxWidth - 3, 0],[maxWidth - 3, maxHeight - 3],[0, maxHeight - 3]],dtype = "float32")
#warping the contour from initial(box) to final(dst) coordinates
M = cv2.getPerspectiveTransform(box_max,dst)
warped = cv2.warpPerspective(imggray, M, (int(maxWidth), int(maxHeight)))

#Blurring the image for better Thresholding
warped = cv2.GaussianBlur(warped, (3,3), 0);
#Thresholding the image using adaptiveThreshloding
th2 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                            cv2.THRESH_BINARY,11,10)

#plotting initial and final images
plt.subplot(121),
plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(th2,cmap = 'gray')
plt.title('final Image'), plt.xticks([]), plt.yticks([])
plt.show()

#Saving various important images along the process into local directory
cv2.imwrite('edge_basic.png',edges)
cv2.imwrite('opening.png',opening)
cv2.imwrite('warped.png',warped)
cv2.imwrite('final.png',th2)
