# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:10:57 2023

@author: user202
"""

import cv2
import matplotlib.pyplot as plt

def image_p(image, scale=1.5, minsize=(224,224)):
    yield image 
    while True:
        w=int(image.shape[1]/scale)
        image=cv2.resize(image, dsize=(w,w))
        if image.shape[0] < minsize[1] or image.shape[1]<minsize[0]:
            break
        yield image  

image_all=cv2.imread('bengal.jpg')
img=image_p(image_all, scale=1.5, minsize=(10,10))
for i, image in enumerate(img):
    print(i)
    if i ==5:
        plt.imshow(image)