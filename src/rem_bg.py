import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import floor as f

def remove_background(img, alpha = 1.5, beta = 0, white = True, resize = False):

    img = img   
    shape  = img.shape[:2]
    landscape = True if shape[1]>shape[0] else False
    if landscape:
        diff = shape[1] - shape[0]
        img = img[:, f(diff/2):f(shape[1] - diff/2)]
    else:
        diff = shape[0] - shape[1]
        img = img[f(diff/2):f(shape[0] - diff/2), :]
    
    img_fin = img
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #blurred = cv2.GaussianBlur(img, (5, 5), 1) # Remove noise
    blurred = img
    
    def edgedetect (channel):
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)
    
        sobel[sobel > 255] = 255; # Some values seem to go above 255. However RGB channels has to be within 0-255
        return sobel
    
    edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
    
    mean = np.mean(edgeImg) * 1
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0
    
    def findSignificantContours (img, edgeImg):
        contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Find level 1 contours
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)
        
            # From among them, find the contours with large surface area.
        significant = []
        tooSmall = edgeImg.size * 5/100 # If contour isn't covering n% of total area of image then it probably is too small, here n = 5
        for tupl in level1:
            contour = contours[tupl[0]]
            area = cv2.contourArea(contour)
            if area > tooSmall:
                significant.append([contour, area])
    
                # Draw the contour on the original image
                cv2.drawContours(img, [contour], 0, (0,0,0),2, cv2.LINE_AA, maxLevel=1)
    
        significant.sort(key=lambda x: x[1])
        #print ([x[1] for x in significant]);
        return [x[0] for x in significant]
    
    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    
    # Find contours
    significant = findSignificantContours(img, edgeImg_8u)
    
    # Mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)
    
    #Finally remove the background
    if white:
        img_fin[mask] = [1, 177, 64]
    else:
        img_fin[mask] = 0
    
    if resize:
        img_fin = cv2.resize(img_fin, (100,100))
    
    return img_fin
