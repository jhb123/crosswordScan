#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:00:17 2022

@author: josephbriggs
"""

import random as rng
import cv2


def show_contours(img, contours, idxs):
    '''
    takes the ouput of cv2.findContours and draws the ones specified by
    a list of indexes over an image. This uses random colours.
    '''
    for i in idxs:
        colour = (rng.randint(0, 256), rng.randint(
            0, 256), rng.randint(0, 256))
        cv2.drawContours(img, contours, i, colour, 2)
    cv2.imshow("show ontours", img)
    cv2.waitKey()


def show_all_contours(img, contours):
    '''
    takes the ouput of cv2.findContours and draws them over an image with
    random colours.
    '''

    # N.B, due to how draw contours works, enumerate is no better here (don't
    # need individual contours)
    for i in range(len(contours)):
        colour = (rng.randint(0, 256), rng.randint(
            0, 256), rng.randint(0, 256))
        cv2.drawContours(img, contours, i, colour, 2)
    cv2.imshow("show all contours", img)
    cv2.waitKey()


def show_all_bounding_boxes(img, contours):
    '''
    takes the ouput of cv2.findContours and draws their bounding boxes
    over an image with random colours.
    '''
    # N.B, due to how draw contours works, enumerate is no better here (don't
    # need individual contours)
    for i in range(len(contours)):
        colour = (rng.randint(0, 256), rng.randint(
            0, 256), rng.randint(0, 256))

        x_coord, y_coord, width, hieght = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x_coord, y_coord),
                      (x_coord+width, y_coord+hieght), colour, 2)

    cv2.imshow("show all contours", img)
    cv2.waitKey()
