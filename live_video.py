# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:30:22 2021

@author: prajw
"""

from find_mask import masker, masker_np
from utils import trim_masks, to_rle, display, convert_to_mask, crop_by_id, convert_color
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

IMAGE_SIZE = 512

with open(".data/label_descriptions.json") as f:
    label_descriptions = json.load(f)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
  
while(True):
      

    ret, frame = vid.read()

    try:
        img, r = masker_np(frame)
        masks, rois = convert_to_mask(img, r)
        idx = list(r['class_ids']).index(2)
        res = crop_by_id(img, masks, idx)
        final_img = convert_color(img, res, [244,196,48])
        img[:,:,0] = np.where(masks[:,:,idx], np.zeros((512,512)), img[:,:,0])
        img[:,:,1] = np.where(masks[:,:,idx], np.zeros((512,512)), img[:,:,1])
        img[:,:,2] = np.where(masks[:,:,idx], np.zeros((512,512)), img[:,:,2])
        final_image = cv2.addWeighted(img,1,final_img,1,0)
        cv2.imshow('frame', cv2.resize(final_image, (1200,800)))
    except:
        cv2.imshow('frame', cv2.resize(frame, (1200,800)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
