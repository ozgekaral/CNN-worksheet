# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:06:31 2023

@author: user202
"""

import cv2
import os
import numpy as np

classes=['bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'train']
colors=np.random.uniform(0, 255, size=(len(classes),3))
n=cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

f=os.listdir()
img_list=[]
for i in f:
    if i.endswith('.jpg'):
        img_list.append(i)
for j in img_list:
    img=cv2.imread(j)
    (h,w)=img.shape[:2]
    a=cv2.dnn.blobFromImage(cv2.resize(img, (400,400)), 0.007843, (300,300), 127.5)
    net.setInput(a)
    detection=net.forward()
    
for z in np.arange(0, detection.shape[2]):
    confidence==detection[0,0,z,2]
    
    if confidence > 0.3:
        idx=int(detection[0,0,z,1])
        box=detection[0,0,z,3:7]*np.array([w,h,w,h])
        (startX, startY, endX, endY)=box.astype('int')
        
        label='{}:{}'.format(classes[idx], confidence)
        cv2.rectangle(img, (startX, startY), (endX, endY), colors[idx], 2)
        y=startY-16 if startY -16>15 else startY+16
        cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
cv2.imshow('ssd', img)
if cv2.waitKey(0) & 0xFF == ord('q'): continue