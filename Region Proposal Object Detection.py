# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:12:11 2023

@author: user202
"""

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import pandas as pd
import cv2

from max_suppression import max_suppression

def selective(img):
    s=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    s.setBaseImage(img)
    s.switchToSelectionSearchQuality()
    r=s.process()
    
    return s[:500]
#Create the model

model=ResNet50(weights='imagenet')
img=cv2.imread('seach.jpg')
img=cv2.resize(img, dsize=(300,300))
r=selective_search(img)

prob=[]
box=[]
for (x, y, w, h) in r:
    if w/float(W)<0.1 or h/float(H)<0.1:
        continue
    roi=image[y:y+h, x:x+w]
    roi=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi=cv2.resize(roi, (224,224))
    
    roi=img_to_array(roi)
    roi=preprocess_input(roi)
    
    prob.append(roi)
    box.append(x,y,w,h)
    
prob=np.array(prop)

#Predict

pred=model.predict(prob)
pred=imagenet_utils.decode_predictions(pred, top=1)

labels={}
min_co=0.8
for (i, j) in enumerate(pred):
    (-, label, prob)=j[0]
    if prob>=min_co:
        (x,y,w,h)=box[i]
        L=labels.get(label, [])
        L.append((box,prob))
        labels[label]=L
        
#Visualization
for label in labels.keys():
    copy_i=image.copy()
    for (box, prob) in labels[label]:
        box=np.array([j[0] for j in labels[label]])
        prob=np.array([j[1] for j in labels[label]])
        box=max_suppression(box,prob)
        
        for ((startX, startY, endX, endY)) in box:
        cv2.rectangle(copy, (startX, startY), (endX, endY), (0,255,0),2)
        y=startY-10 if  startY-10>10 else startY + 10
        
    cv2.imshow('photo', copy_i)
    
        
        
        
        
    


    
    