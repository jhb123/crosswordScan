#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:40:01 2022

@author: josephbriggs
"""

import random as rng
import string

from PIL import Image, ImageFont, ImageDraw 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_all_contours(img,contours):
    for i in range(len(contours)):
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        cv2.drawContours(img,contours,i,colour,2)
    cv2.imshow("show all contours",img)
    cv2.waitKey()
    
if __name__ == "__main__":
    path_to_font = r'/System/Library/Fonts/Supplemental/Arial.ttf'
    # specified font size
    font = ImageFont.truetype(path_to_font, 20) 
      
    letters = string.ascii_letters+string.punctuation+'1234567890'
    
    for letter in letters:
        image = Image.new("1",(20,25),0)
    
        # drawing text size
        draw = ImageDraw.Draw(image) 
    
        draw.text((2, 0), letter, font = font, fill ="white", align ="center") 
        
        np_image = np.array(image,dtype = int)
        filled_idx = np.argwhere(np_image==1)
        
        x1 = filled_idx[:,0].min()
        x2 = filled_idx[:,0].max()
        y1 = filled_idx[:,1].min()
        y2 = filled_idx[:,1].max()
        
        
        fig,ax = plt.subplots()
        ax.imshow(np_image[x1:x2+1,y1:y2+1])
        
