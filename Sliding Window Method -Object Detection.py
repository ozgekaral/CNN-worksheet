# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:31:32 2023

@author: user202
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def window(img, s, size):
    for y in range(0 ,img.shape[0]-size[1], s):
        for x in range(0,img.shape[1]-size[0], s):
            
            yield(x,y, img[y:y+s[1], x:x+s[0]])
            
image=cv2.imread('bengal.jpg')
im_func=window(image, 10, (100,100))
for i,img in enumerate(im_func):
    print(i)