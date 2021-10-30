# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:50:06 2021

@author: prajw
"""

from find_mask import masker, masker_np
from utils import trim_masks, to_rle, display, convert_to_mask, crop_by_id, rgb_to_hsv, convert_color, tensor_to_image
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

IMAGE_SIZE = 512

with open(".data/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

img, r = masker('s4.jpg')

masks, rois = convert_to_mask(img, r)

rgb = list(map(int, input("Enter rgb").split()))    
#rgb = [0,255,0]
for idx, i in enumerate(r['class_ids']):
    print(idx, i, label_names[i-2])


# idx = list(r['class_ids']).index(2)
images = []
idx_s = []
while True:
    idx = input()
    if idx == 'q':
        break
    idx_s.append(int(idx))
    idx = int(idx)    
    res = crop_by_id(img, masks, idx)
    
    # display(res)
    final_image = img.copy()

    
    final_img = convert_color(img, res, rgb)
    images.append(final_img)
    print('enter q to break, else enter index \n')
'''
final_image = img.copy()
final_image[:,:,0] = np.where(masks[:,:,4], np.zeros((512,512)), final_image[:,:,0])
final_image[:,:,1] = np.where(masks[:,:,4], np.zeros((512,512)), final_image[:,:,1])
final_image[:,:,2] = np.where(masks[:,:,4], np.zeros((512,512)), final_image[:,:,2])

display(final_image)
'''

l = len(images)

if l ==1:
    for idx, i in enumerate(images):
        final_image[:,:,0] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,0])
        final_image[:,:,1] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,1])
        final_image[:,:,2] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,2])
        
        final_image = cv2.addWeighted(final_image,1,i,1,0)
else:
    for idx, i in enumerate(images):
        final_image[:,:,0] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,0])
        final_image[:,:,1] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,1])
        final_image[:,:,2] = np.where(masks[:,:,idx_s[idx]], np.zeros((512,512)), final_image[:,:,2])
        
        final_image = cv2.addWeighted(final_image,1,i,1/2*l,0)
    
comp = cv2.hconcat([img, final_image])
display(comp)
cv2.imwrite('test.jpg', comp)

#%%
