#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:03:24 2022

@author: josephbriggs
"""

import cv2
import numpy as np


def find_crossword_contour(img):
    '''


    Parameters
    ----------
    img : numpy array
        an image loaded in with cv2.imread.

    Returns
    -------
    contour : numpy array.
        The opencv222 contour that surrounds the crossword

    '''

    # pre-process image
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gs_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 101, 2)

    # find the area of each contour
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.zeros(len(contours))
    for i, contour in enumerate(contours):
        areas[i] = cv2.contourArea(contour)

    # assume image takes up 1/4 of screen
    idxs = np.where(areas > thresh.size/16)[0]

    # find the squarest contour.
    score = 1e6
    for i in idxs:
        _, _, width, height = cv2.boundingRect(contours[i])
        aspect = width/height
        if np.abs(1-aspect) < score:
            score = np.abs(1-aspect)
            cw_bbox_idx = i

    return contours[cw_bbox_idx]


def main():
    '''
    demo of the grid extraction

    Returns
    -------
    None.

    '''

    img = cv2.imread('crossword1.jpeg')
    cword_contour = find_crossword_contour(img)

    x, y, width, height = cv2.boundingRect(cword_contour)
    rect = cv2.minAreaRect(cword_contour)

    cv2.drawContours(img, cword_contour, -1, (255, 255, 0), 4)
    cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 3)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    crossword = img[y:y+height, x:x+width]

    cv2.imshow('Contours', crossword)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
