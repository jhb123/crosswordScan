#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:00:17 2022

@author: josephbriggs
"""

import random as rng
import cv2

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