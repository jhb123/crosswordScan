#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:05:52 2022

@author: josephbriggs
"""
import importlib.resources
import random as rng
import cv2
import numpy as np

def show_contours(img,contours,idxs):
    for i in idxs:
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        cv2.drawContours(img,contours,i,colour,2)
    cv2.imshow("cshow ontours",img)
    cv2.waitKey()
    
def show_all_contours(img,contours):
    for i in range(len(contours)):
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        cv2.drawContours(img,contours,i,colour,2)
    cv2.imshow("show all contours",img)
    cv2.waitKey()



if __name__ == "__main__":
    test_image = "crossword1.jpeg"
    crossword_location = "cws.resources.crosswords"
    
    with importlib.resources.path(crossword_location,test_image) as path:
        input_image = cv2.imread(str(path))
    
    gs_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gs_img, (5, 5),1)
    
    edges = cv2.Canny(blur,100,200,False)
    # kernel = np.ones((3,3))
    # dilate = cv2.dilate(edges,kernel,1)
    
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(edges, contours, -1, 255, 11)
    # show_all_contours(input_image,contours)
    
    
    h, w = edges.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(edges, mask, (0,0), 123)
    filled = cv2.inRange(edges, 122, 124)
    
    cv2.imshow("edges",filled)
    cv2.waitKey()
# minLineLength = 10
# maxLineGap = 20


# lines = cv2.HoughLinesP(dilate,1,np.pi/180,10,minLineLength=minLineLength,maxLineGap=maxLineGap)

# text_location = np.zeros(input_image.shape,dtype = input_image.dtype)

# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
        
#         if np.abs((y2-y1)/(x2-x1)) < 1:
        
#             cv2.line(text_location,(x1,y1),(x2,y2),(255,255,255),2)


# gs_img = cv2.cvtColor(text_location, cv2.COLOR_BGR2GRAY)
# _,thresh = cv2.threshold(gs_img,50,255,cv2.THRESH_BINARY)

# dilate = cv2.dilate(thresh,np.ones((11,11)),1)



# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9,2)))


# contours, _ = cv2.findContours(
#     dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# show_all_contours(input_image,contours)