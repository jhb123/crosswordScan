#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:05:52 2022

@author: josephbriggs
"""
# import random as rng
import importlib.resources
import cv2
import numpy as np
# import cws.utils as utils
import cws.grid_extract 
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import pytesseract
# from pytesseract import Output
import string
# from cv2 import dnn_superres

def flatten_text_box(img,contour):
    
    x,y,w,h = cv2.boundingRect(contour)
    
    
    # cv2.imshow("some_text",img[y:y+h,x:x+w])
    # cv2.waitKey()
    
    return img[y:y+h,x:x+w]

def locate_text_boxes(input_image,show = False):
    
    rows, cols, _channels = map(int, input_image.shape)
    
    cross_word_contour = cws.grid_extract.find_crossword_contour(input_image)
    
    gs_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    cv2.fillPoly(gs_img, [cross_word_contour], [255,255,255])
    
    img_for_reading = cv2.adaptiveThreshold(gs_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,51,20)
    
    # img_for_reading = cv2.erode(img_for_reading,np.ones((3,3)),1)
    
    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img_for_reading) # also include any config options you use
    boxes_list = boxes.splitlines()
    
    h, w = img_for_reading.shape # assumes color image
    blacklist = string.punctuation
    # draw the bounding boxes on the image
    blank = np.zeros(img_for_reading.shape,dtype = np.uint8)
    
    boxes_list = [b for b in boxes_list if b[0] not in blacklist]
    
    for box in boxes_list:
        b = box.split(" ")
        x1 = int(b[1])
        x2 = int(b[3])
        y1 =  h - int(b[2])
        y2 = h  - int(b[4])
        
        area = (x2-x1)*(y1-y2)
        if area < 1000:
            cv2.rectangle(blank,
                            (x1, y1), 
                            (x2, y2),
                            255, -1)
            
    dilate = cv2.dilate(blank,np.ones((21,21)),3)
    
    contours, hierarchy = cv2.findContours(dilate, 
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    
    
    if show:
        for i,contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            epsilon = 0.05*cv2.arcLength(hull, True)
        
            approx = cv2.approxPolyDP(hull, epsilon, True)
            M = cv2.moments(contour)
        
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            cv2.putText(input_image, 
                str(i), 
                (cx, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, 
                (0, 255, 255), 
                3, 
                cv2.LINE_4)
        
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(input_image,[box],0,(255,0,255),2)
            cv2.drawContours(input_image, [contour], -1, (0, 0, 255),  2)
            cv2.drawContours(input_image, [hull], -1, (0, 255, 0),  2)
            cv2.drawContours(input_image, approx, -1, (255, 0, 0),  10)
            
        cv2.imshow("text_boxes",input_image)
        cv2.waitKey()
            
    return contours

def process_text_box(img):
    
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = map(int, gs_img.shape)
  
    norm_image = cv2.normalize(gs_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    img_for_reading = cv2.adaptiveThreshold(norm_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,31,30)
    
    erode = cv2.erode(img_for_reading,np.ones ((2,2)),1)
    print(pytesseract.image_to_string(img_for_reading))
    
    cv2.imshow("image for reading",img_for_reading )
    cv2.waitKey()
               

    
def main():
    '''
    clue extraction place holder and experiments
    '''

    test_image = "crossword3.jpeg"
    crossword_location = "cws.resources.crosswords"
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract'

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))
 
   
    contours = locate_text_boxes(input_image,show = False)
    text_box = flatten_text_box(input_image,contours[4])
    process_text_box(text_box)


#     # data = list(zip(xs, ys,aspects,areas,bb_areas))
    
#     # elbow_method(data)

#     # num_clusters = 9
#     # kmeans = KMeans(n_clusters=num_clusters)
#     # kmeans.fit(data)
    
#     # max_groups = kmeans.labels_.max()
#     # for i in range(max_groups):
#     #     colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        
#     #     idxs = np.squeeze(np.argwhere(kmeans.labels_==i))

#     #     for i in idxs:
#     #         cv2.drawContours(input_image,contours,i,colour,2)

#         # utils.show_all_bounding_boxes(input_image, contours)

#     # print(pytesseract.image_to_string(thresh))

#     # cv2.imshow("thresh",img_for_reading)
#     # cv2.waitKey()
    

    
# def elbow_method(data):
    
#     inertias = []
#     num_test_points = 25
#     for i in range(1,num_test_points):
#         kmeans = KMeans(n_clusters=i)
#         kmeans.fit(data)
#         inertias.append(kmeans.inertia_)
#     fig,ax = plt.subplots()
#     ax.plot(range(1,num_test_points), inertias, marker='o')
#     ax.set_title('Elbow method')
#     ax.set_xlabel('Number of clusters')
#     ax.set_ylabel('Inertia')
    

if __name__ == "__main__":
    main()
