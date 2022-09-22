#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:05:52 2022

@author: josephbriggs
"""
import random as rng
import importlib.resources
import cv2
import numpy as np
import cws.utils as utils
import cws.grid_extract 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    '''
    clue extraction place holder and experiments
    '''

    test_image = "crossword2.jpeg"
    crossword_location = "cws.resources.crosswords"

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    cross_word_contour = cws.grid_extract.find_crossword_contour(input_image)
    
    gs_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    cv2.fillPoly(gs_img, [cross_word_contour], [255,255,255])
    
    blur = cv2.GaussianBlur(gs_img, (11,11), 1)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,5,10)
    

    lines = cv2.HoughLinesP(thresh,
                            4,
                            np.pi/180,
                            10,
                            minLineLength=20,
                            maxLineGap=20)

    line_lengths = np.zeros(len(lines))
    line_grads = np.zeros(len(lines))

    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
        line_lengths[i] = np.sqrt( (x2-x1)**2 + (y2-y1)**2)
        line_grads[i] = np.abs( (y2-y1)/(x2-x1))

    line_std = np.std(line_lengths)
    line_mean = np.std(line_lengths)

    filtered_lines = np.argwhere(np.logical_and(
            line_lengths < line_mean+2*line_std,
            line_grads < 0.5))
    
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line[0]

        if np.isin(i,filtered_lines):
            cv2.line(thresh,(x1,y1),(x2,y2),255,2)
        # else:
        #     cv2.line(thresh,(x1,y1),(x2,y2),0,2)

    filtered_lines = np.argwhere(np.logical_and(
            line_lengths > line_mean+2*line_std,
            line_grads > 10))
       
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line[0]

        if np.isin(i,filtered_lines):
            cv2.line(thresh,(x1,y1),(x2,y2),0,20)
    
        
    dilate = cv2.dilate(thresh,np.ones((5,5)),3)

    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("edges", thresh)
    # cv2.waitKey()

    # utils.show_all_bounding_boxes(input_image, contours)

    # # x = np.zeros(len(contours))
    # # y = np.zeros(len(contours))
    # # r = np.zeros(len(contours))
    widths = np.zeros(len(contours))
    heights = np.zeros(len(contours))
    xs = np.zeros(len(contours))
    ys = np.zeros(len(contours))
    areas = np.zeros(len(contours))
    solidities = np.zeros(len(contours))
    extents = np.zeros(len(contours))
    bb_areas = np.zeros(len(contours))
    aspects = np.zeros(len(contours))
    
    for i,c in enumerate(contours):

        x,y,w,h = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        hull = cv2.convexHull(contours[i])
        hull_area = cv2.contourArea(hull)
        rect_area = w*h
        solidity = float(area)/hull_area
        extent = float(area)/rect_area
        aspect = w/h
        
        
        widths[i] = w
        heights[i] = h
        xs[i] = x
        ys[i] = y
        areas[i] = area
        solidities[i] = solidity
        extents[i] = extent
        bb_areas[i] = rect_area
        aspects[i] = aspect
        # centre, radius = cv2.minEnclosingCircle(c) 
        # x[i] = centre[0]
        # y[i] = centre[1]
        # r[i] = radius
            
    # data = list(zip(widths, heights, xs, ys,areas))
    xs = xs/xs.max()
    ys = ys/ys.max()
    widths = widths/widths.max()
    heights = heights/heights.max()
    bb_areas = bb_areas/bb_areas.max()
    aspects =aspects/aspects.max()
    areas = areas/areas.max()

    data = list(zip(xs, ys,aspects))

    elbow_method(data)

    num_clusters = 7
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    for i in range(num_clusters):
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        
        idxs = np.squeeze(np.argwhere(kmeans.labels_==i))
        try:
            for i in idxs:
                cv2.drawContours(input_image,contours,i,colour,2)
        except:
            pass

    cv2.imshow("show all contours",input_image)
    cv2.waitKey()

    # cv2.imshow("edges", dilate)
    # cv2.waitKey()


def elbow_method(data):
    
    inertias = []
    num_test_points = 25
    for i in range(1,num_test_points):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    fig,ax = plt.subplots()
    ax.plot(range(1,num_test_points), inertias, marker='o')
    ax.set_title('Elbow method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    
if __name__ == "__main__":
    main()
