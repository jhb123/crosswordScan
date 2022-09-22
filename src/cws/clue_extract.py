#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:05:52 2022

@author: josephbriggs
"""
import importlib.resources
import cv2
import numpy as np


def main():
    '''
    clue extraction place holder and experiments
    '''

    test_image = "crossword1.jpeg"
    crossword_location = "cws.resources.crosswords"

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    gs_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gs_img, (5, 5), 1)

    edges = cv2.Canny(blur, 100, 200, False)
    # kernel = np.ones((3,3))
    # dilate = cv2.dilate(edges,kernel,1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(edges, contours, -1, 255, 11)
    # show_all_contours(input_image,contours)

    height, width = edges.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(edges, mask, (0, 0), 123)
    filled = cv2.inRange(edges, 122, 124)

    cv2.imshow("edges", filled)
    cv2.waitKey()


if __name__ == "__main__":
    main()
