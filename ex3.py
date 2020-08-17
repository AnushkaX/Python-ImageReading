import cv2
import numpy as np

img =cv2.imread('bacteria.tiff')

cv2.imshow('Original',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#thresholding
t,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Thesh',thresh)

#fill holes
kernel = np.ones((3,3),np.uint8)
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
cv2.imshow('Close', close)

#background
sure_bg = cv2.dilate(close, kernel, iterations=1)
cv2.imshow('Sure Background',sure_bg)

#foreground
dist=cv2.distanceTransform(close,cv2.DIST_L1,3)
t,sure_fg=cv2.threshold(dist,dist.max()*0.32,255,0)
sure_fg=np.uint8(sure_fg)
cv2.imshow('Sure Foreground',sure_fg)

unknown=cv2.subtract(sure_bg,sure_fg)

#marker labeling
m,markers=cv2.connectedComponents(sure_fg)
markers=markers+1
markers[unknown==255]=0

#apply watershed algorithm
makers=cv2.watershed(img, markers)
img[markers==-1]=[0, 255, 0]
cv2.imshow('Final result', img)

#counting bacteria cells

th=cv2.inRange(img,(0,255,0),(0,255,0))
cv2.imshow('th',th)
th=cv2.bitwise_not(th)
cv2.imshow('th',th)

print('Number of bacteria cells : ',m)


