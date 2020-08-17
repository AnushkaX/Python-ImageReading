import cv2
import numpy as np

img=cv2.imread('overlap_coins.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
t,thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Threshed', thresh)

#sure background
kernel=np.ones((3,3), np.uint8)
sure_bg=cv2.dilate(thresh, kernel,iterations = 2)
cv2.imshow('Sure Background', sure_bg)

#sure foreground
dist_transform=cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
t,sure_fg=cv2.threshold(dist_transform,0.5*dist_transform.max(), 255, 0)
cv2.imshow('Sure Foreground',sure_fg)

#unknown boundry
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg, sure_fg)
cv2.imshow('Unknown', unknown)

#marker lebeling
ret,markers=cv2.connectedComponents(sure_fg)
markers=markers+1
markers[unknown==255]=0

watershed_img=cv2.watershed(img, markers)

img[watershed_img == -1]=[0, 255,0]
cv2.imshow('Final', img)

print('Number of coins', ret)
