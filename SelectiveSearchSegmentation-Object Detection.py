# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:03:24 2023

@author: user202
"""

import cv2
import random

img=cv2.imread('search.jpg')
img=cv2.resize(img, dsize=(600,600))
cv2.imshow('search', img)

s=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
s.setBaseImage(img)
s.switchToSelectionSearchQuality()
r=s.process()

for (x,y,w,h) in r[:50]:
    c=[random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(r, (x,y), (x+w,y+h), color, 2)
cv2.imshow('r', r)



    
