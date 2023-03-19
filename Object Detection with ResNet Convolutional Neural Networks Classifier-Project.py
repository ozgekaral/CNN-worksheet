# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:51:41 2023

@author: user202
"""

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import cv2


from max_suppression import max_suppression
from image_p import image_p
from window import window


WIDTH=600
HEIGHT=600  
PYR_SCLE=1.5
WIN_STEP=16
ROI_SIZE=(200,150)
INPUT_SIZE=(224,224)

#Create model

model=ResNet50(weights='imagenet', include_top=True)  

o_img=cv2.imread('bengal.jpg')
o_img=cv2.resize(o_img, dsize=(WIDHT, HEIGHT))

(H,W)=o_img.shape(:2)

#1-Image pyramid
pyramid=image_p(o_img, PYR_SCLE, ROI_SIZE)

rois=[]
locs=[]

for image in p:
    scale=w(float()image.shape[1])
    for x,y,roioimg in window(image, WIN_STEP, ROI_SIZE):
        x=int(x*scale)
        y=int(y*scale)
        w=int(ROI_SIZE[0]*scale)
        h=w=int(ROI_SIZE[1]*scale)
        
        roi=cv2.resize(roioimg, INPUT_SIZE)
        roi=img_to_array(roi)
        roi=preprocess_input(roi)
        
rois=np.array(rois, dtype='float32')
#Predict

pred=model.predict(rois)

imagenet_utils.decode_predictions(pred, top1=1)

#Classification
labels={}
min_co=0.9

for (i,j) in enumerate(pred):
    (-, label, prob)=j[0]
    if prob>=min_co:
        box=locs[i]
        L=labels.get(label, [])
        L.append((box,prob))
        labels[label]=L

for label in labels.keys():
    copy_i=o_img.copy()

#draw box 
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY)=box 
        cv2.rectangle(copy, (startX, startY), (endX, endY), (0,255,0),2)
    cv2.imshow('photo', copy_i)        
    