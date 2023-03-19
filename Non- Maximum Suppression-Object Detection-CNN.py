# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:01:45 2023

@author: user202
"""

import cv2
import numpy as np
import pandas as pd

def max_suppression(box, prob=None, overlabThress=0,3):
    if len(box)==0:
        return
    if box.dtype.kind=='i':
        box=box.astype('float')
    
    x1=box[:,0]
    x2=box[:,2]
    y1=box[:,1]
    y2=box[:,3]
    
    area=(x2-x1+1)*(y2-y1+1)
    idxs=y2
    
    if prob is not None:
        idxs=prob  
        
    idxs=np.argsort(idxs)
    
    pick=[]
    
    while len(idxs)>0:
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        
        xx1=np.maximum(x1[i], x1[idxs[:last]])
        xx2=np.maximum(x1[i], x1[idxs[:last]])
        yy1=np.maximum(x1[i], x1[idxs[:last]])
        yy2=np.maximum(x1[i], x1[idxs[:last]])
        
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        
        overlap=(w*h)/area[idxs[:last]]
        idxs=np.delete(idxs, np.concatenate(([last],np.where(overlap>overlapThresh))))
        
        return box[pick].astype('int')
        
    
    