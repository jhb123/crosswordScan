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
import cws.utils as utils
import cws.grid_extract 
# from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import random as rng

import matplotlib.pyplot as plt
import pytesseract
# from pytesseract import Output
import string
# from cv2 import dnn_superres
import re



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
        if area < 2000:
            cv2.rectangle(blank,
                            (x1, y1), 
                            (x2, y2),
                            255, -1)
            
    dilate = cv2.dilate(blank,np.ones((51,51)),3)
    
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
    
    # erode = cv2.erode(img_for_reading,np.ones ((2,2)),1)
    
    return img_for_reading
    # print(pytesseract.image_to_string(img_for_reading))
    
    # cv2.imshow("image for reading",img_for_reading )
    # cv2.waitKey()
               
def match_template(arr,pattern):
    matched_template = cv2.matchTemplate(arr.astype('uint8'),pattern.astype('uint8'),cv2.TM_SQDIFF)
    matched_template = matched_template.ravel()
    return matched_template

        
def segment_page_preprocess(img):
    
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_for_reading = cv2.adaptiveThreshold(gs_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,9,15)
    
    # img_for_reading = cv2.erode(img_for_reading,np.ones((3,3)),1)
    whitelist = string.ascii_letters + '1234567890'

    configs = ["--psm 3",
               "-c tessedit_char_whitelist={whitelist}"]
    cfg_str = " ".join(configs)
    
    data = pytesseract.image_to_data(img_for_reading,config=cfg_str) # also include any config options you use
    lefts = [re.split('\t',d)[6] for d in data.splitlines()[1:]]
    tops = [re.split('\t',d)[7] for d in data.splitlines()[1:]]
    widths = [re.split('\t',d)[8] for d in data.splitlines()[1:]]
    heights = [re.split('\t',d)[9] for d in data.splitlines()[1:]]
    texts = [re.split('\t',d)[11] for d in data.splitlines()[1:]]
    confs = [re.split('\t',d)[10] for d in data.splitlines()[1:]]

    word_loc = np.zeros(img_for_reading.shape,dtype = np.uint8)
    
    lefts = np.array([int(i) for i in lefts])
    tops = np.array([int(i) for i in tops])
    widths = np.array([int(i) for i in widths])
    heights = np.array([int(i) for i in heights])
    # confs = np.array([float(i) for i in confs])
    
    heights_for_average = []
    
    w,h = img_for_reading.shape 
    
    whitelist = rf'[\d{string.ascii_letters}]'

    for left,top,width,height,text,conf in zip(lefts,tops,widths,heights,texts,confs):
        
        is_ok = bool(re.search(whitelist, text))
        if is_ok:
            heights_for_average.append(height)
            x1 = left
            x2 = left + width
            y1 = top 
            y2 = height + top 
            
         
            cv2.rectangle(word_loc,
                            (x1, y1), 
                            (x2, y2),
                            255, -1)
            
            # cv2.putText(img, 
            #             text, 
            #             (x1, y2), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            #             (0, 0, 255), 
            #             2, 
            #             cv2.LINE_4)
            
            # cv2.putText(img, 
            #             conf, 
            #             (x1+5, y2), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            #             (0, 255, 255), 
            #             2, 
            #             cv2.LINE_4)
    
    
    

    return word_loc,np.median(heights_for_average)


def segment_page_idxs(word_loc,smoothing = 100,thresh_f = 5):
    
    row_scan = np.sum(word_loc,axis = 1)
    
    scan_smooth_factor = smoothing
    size = int(row_scan.size/scan_smooth_factor)
    if size%2 == 0:
        corr = 1
    else:
        corr = 0
    smoother = np.ones(size + corr)/(size+corr)
    row_scan_s = np.convolve(row_scan,smoother ,mode = 'valid')

    # thresh = row_scan.size/thesh_f #% of pixels
    # print(thresh)
    thresh = np.percentile(row_scan_s, [thresh_f])
    # print(thresh)
    
    white_spaces = np.argwhere(row_scan_s <= thresh)
    
    # fig,ax = plt.subplots()
    # ax.plot(row_scan)
    # ax.plot(row_scan_s)
    # ax.scatter(white_spaces,row_scan[white_spaces],zorder = 16,color = 'k')

    return white_spaces
           
def segment_page(word_loc):
    x_idxs = segment_page_idxs(word_loc,
                               smoothing = 100,thresh_f = 10)
    y_idxs = segment_page_idxs(np.swapaxes(word_loc, 0, 1),
                               smoothing = 100,thresh_f = 10)
    
    return x_idxs,y_idxs

    
def get_text_boxes(img):
    
    x_idxs,y_idxs = segment_page(img)
    w,h = img.shape
    
    text_boxes_mask = 255*np.ones((w,h),dtype = np.uint8)
    
    text_boxes_mask[x_idxs,:] = 0
    text_boxes_mask[:,y_idxs] = 0
    
    
    totalLabels,labels,values, centroid  = cv2.connectedComponentsWithStats(text_boxes_mask, 8)

    return totalLabels,labels, values, centroid

def get_text_box_idx(img,labels,idx,pad):
    
    mask = labels == idx
    
    mask = 255*mask.astype(np.uint8)
    
    kernel = np.ones((pad,pad))

    mask_dilate = cv2.dilate(mask,kernel,1)
    
    # args = np.argwhere(mask_dilate>0)
    
    # x1 = np.argmin(mask_dilate,axis=1) #np.min(args,axis = 0)
    # x2 = np.max(args,axis = 0)
    # y1 = np.min(args,axis = 1)
    # y2 = np.max(args,axis = 1)
    # print(f'{x1=}')
    # print(f'{y1=}')
    # print(f'{x2=}')
    # print(f'{y2=}')
    
    mask_array = np.ix_(mask_dilate.any(1), mask_dilate.any(0))

    # print(args)
    # box_blur = cv2.filter2D(mask, -1, kernel) == 1
    
    return img[mask_array]

    
def text_box_pre_process(img,word_height):
    

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "ESPCN_x4.pb"
    sr.readModel(path)
    sr.setModel("espcn",4)
    result = sr.upsample(img)
        
    gs_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    
    size_factor = int(word_height*4) + 1
    # # if word_height > 40:
    img_for_reading = cv2.medianBlur(gs_img, 5)

    img_for_reading = cv2.adaptiveThreshold(img_for_reading,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,201,20)
    
    # kernel = np.ones((5,5))
    # img_for_reading = cv2.morphologyEx(img_for_reading, cv2.MORPH_CLOSE, kernel)
    # blur = cv2.GaussianBlur(img_for_reading,(9,9),1)
        
    cv2.imshow("text to analyse",img_for_reading)
    cv2.waitKey()
    
    return img_for_reading

def text_box_clue_extraction(img):
    # print(pytesseract.image_to_osd(img))
    # try --psm 3
    all_text = pytesseract.image_to_string(img,config='--psm 3  --user-patterns my.patterns')
    raw_text = all_text
    # print(all_text)
    left_brackets = "[{\[]"
    right_brackets = "[\|}\]]"
    all_text = re.sub(left_brackets,'(',all_text)
    all_text = re.sub(right_brackets,')',all_text)
    all_text = all_text.replace('\n',' ')
    print(f"{'':@^30}")
    print(all_text)    
    
    pattern = r'(\(?[\d.\-,gsGS\s]*\))|(\([\d.\-,gsGS\s]*\)?)'
    
    split_text = re.split(pattern, all_text)
    split_text = [s for s in split_text if s != '']
    split_text = [s for s in split_text if s != None ]
    split_text = [s for s in split_text if s != ' ' ]

    print(split_text)
    if len(split_text) > 1:
        
        clues = split_text[::2]
        word_lengths = split_text[1::2]
        clue_lengths = split_text[1::2]

    else:
        clues = []
        word_lengths = []
        clue_lengths = []
    print('--')
    print(f"{clues=}")
    print('--')
    print(f"{word_lengths=}")
    print('--')
    print(f"{clue_lengths=}")
    
    return clues,word_lengths,clue_lengths,raw_text


    
def show_box_areas(totalLabels,labels):
    
    debug_boxes = np.zeros((labels.shape[0],labels.shape[1],3),dtype = np.uint8 )

    for i in range(totalLabels):
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        debug_boxes[labels == i] = colour
    
    cv2.imshow("text boxes",debug_boxes)
    cv2.waitKey()
    
def show_box_areas_over_img(totalLabels,stats,centre,img):
    img_to_label = np.copy(img)
    
    for i in range(totalLabels):
        colour = (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256))
        x1 = stats[i,0]
        x2 = x1+stats[i,2]
        y1 = stats[i,1]
        y2 = y1+stats[i,3]
        
        cv2.rectangle(img_to_label,
                        (x1, y1), 
                        (x2, y2),
                        colour, 5)
        cv2.putText(img_to_label, 
                    str(i), 
                    (int(centre[i][0]), int(centre[i][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 
                    colour, 
                    10, 
                    cv2.LINE_4)
    
    cv2.imshow("text boxes",img_to_label)
    cv2.waitKey()

def text_box_extraction_pipeline(input_image):
            
    # remove grid
    cross_word_contour = cws.grid_extract.get_grid_contour_by_blobbing(input_image)    
    cv2.fillPoly(input_image, [cross_word_contour], [255,255,255])
        
    word_loc,word_height = segment_page_preprocess(input_image)
    # cv2.imshow("word location coarse",word_loc)
    # cv2.waitKey()
    
    totalLabels,labels,stats,centroid = get_text_boxes(word_loc)
    # show_box_areas(totalLabels,labels)
    # show_box_areas_over_img(totalLabels,stats,centroid,input_image)
    # for i in range(1,totalLabels):
    #     get_text_box_idx(input_image,labels,i,50)
    
    all_clues = []
    all_word_lengths = []
    all_clue_lengths = []
    
    for i in range(1,totalLabels):
        cropped_text_box = get_text_box_idx(input_image,labels,i,1)
        cropped_text_box_pre_processed = text_box_pre_process(cropped_text_box,word_height)


        clues,word_lengths,clue_lengths,raw_text = text_box_clue_extraction(cropped_text_box_pre_processed)
        
        all_clues = all_clues+clues
        all_word_lengths = all_word_lengths + word_lengths
        all_clue_lengths = all_clue_lengths + clue_lengths
        

    for c,w,l in zip(all_clues,all_word_lengths,all_clue_lengths):
        print(f'{c.strip()} :: {w}\n')
            
    return all_clues,all_word_lengths,all_clue_lengths

def main():
    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract'

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))
        
    text_box_extraction_pipeline(input_image)


if __name__ == "__main__":
    main()
