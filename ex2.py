import cv2
import numpy as np

img=cv2.imread('red bc (1).jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

#thresholding
t,thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('thresh', thresh)

#sure background
kernel = np.ones((5,5),np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations = 2)
cv2.imshow('Sure Background', sure_bg)

#sure foreground
erode = cv2.erode(thresh, kernel, iterations = 1)
dist = cv2.distanceTransform(erode, cv2.DIST_L1, 5)
ret, sure_fg = cv2.threshold(dist,dist.max()*0.3, 255, 0)
sure_fg = np.uint8(sure_fg)
cv2.imshow('Sure Foreground', sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

#Watershed
l, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255]=0

water_sheded = cv2.watershed(img, markers)
img[markers==-1]=[255,0,0]
cv2.imshow('water_sheded',img)
print('number of red blood cells :',l )
