import cv2
import numpy as np

img = cv2.imread('cell_segmentation_02.jpg')

cv2.imshow('original image',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

t,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Thresh', thresh)
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)

sure_fg = cv2.erode(opening, kernel, iterations=2)
cv2.imshow('Sure Foreground', sure_fg)

sure_bg = cv2.dilate(opening, kernel, iterations=2)
cv2.imshow('Sure Background', sure_bg)


unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('Unknown', unknown)

ret,markers=cv2.connectedComponents(sure_fg)
markers=markers+1
markers[unknown==255]=0

markers=cv2.watershed(img,markers)
img[markers==-1]=[0,255,0]
cv2.imshow('Cell Segments',img)

#number of cells
print('Number of cells : ',ret)
