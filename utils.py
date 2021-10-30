# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:10:09 2021

@author: prajw
"""

import numpy as np
import itertools
import cv2
import PIL
import tensorflow as tf

def to_rle(bits):
    rle = []
    pos = 1
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, len(group_list)])
        pos += len(group_list)
    return rle


def trim_masks(masks, rois, class_ids):
    class_pos = np.argsort(class_ids)
    class_rle = to_rle(np.sort(class_ids))
    
    pos = 0
    for i, _ in enumerate(class_rle[::2]):
        previous_pos = pos
        pos += class_rle[2*i+1]
        if pos-previous_pos == 1:
            continue 
        mask_indices = class_pos[previous_pos:pos]
        
        union_mask = np.zeros(masks.shape[:-1], dtype=bool)
        for m in mask_indices:
            masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
            union_mask = np.logical_or(masks[:, :, m], union_mask)
        for m in mask_indices:
            mask_pos = np.where(masks[:, :, m]==True)
            if np.any(mask_pos):
                y1, x1 = np.min(mask_pos, axis=1)
                y2, x2 = np.max(mask_pos, axis=1)
                rois[m, :] = [y1, x1, y2, x2]
            
    return masks, rois

def display(img):
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def convert_to_mask(img, r, IMAGE_SIZE = 512):
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = trim_masks(masks, rois, r['class_ids'])
    else:
        masks, rois = r['masks'], r['rois']
    return masks, rois

def crop_by_id(img, masks, processing):
    res = img.copy()    
    res[...,0] = np.where(masks[:,:,processing], img[:,:,0], masks[:,:,processing])
    res[...,1] = np.where(masks[:,:,processing], img[:,:,1], masks[:,:,processing])
    res[...,2] = np.where(masks[:,:,processing], img[:,:,2], masks[:,:,processing])
    return res

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def convert_color(img, res, rgb, IMAGE_SIZE = 512):
    hsv_image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)
    
    
    #Get the resultant HSV values from RGB
    aa=rgb_to_hsv(rgb[0],rgb[1],rgb[2])
        
    #HUE adjustments
    hsv_image[:,:,0] = (np.ones(shape=(IMAGE_SIZE,IMAGE_SIZE))[:,:])*int(aa[0]//2) # Changes the H value
    
    
    #SATURATION adjustments
    hsv_image[:,:,1] = np.ones(shape=(IMAGE_SIZE,IMAGE_SIZE))*int(aa[1]) # Changes the S value
      
    
    #VALUE adjustments
    #value_new = (max(rgb)//255)*100    
    #hsv_image[:,:] = np.where(mask_image==m_rgb,value_new,hsv_image[:,:])
    
    #Change the color of image to BGR from HSV
    bgr_image= cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) 
    
    #combine modified and orignal image
    final_image=cv2.addWeighted(img,0,bgr_image,1,0)
    return final_image

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

@tf.function
def load_img(path_to_img, dim = 512):
  max_dim = dim
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img