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


def crop_to_crossword(img, contour):
    '''
    Prototype crossword cropping and perspective fix. There are some obvious
    improvements to make to this such as removing all the hardcoded hyperparams

    Parameters
    ----------
    img : np.array
        image of a crossword.
    contour : np.array
        a contour which encloses the crossword.

    Returns
    -------
    warped_img : np.array
        a 510*510 version of the crossword i.e. cropped and perspective fixed.

    '''
    # if the points in the contour are very close, the calcultion of angle is
    # not very good
    contour = contour[::2]
    angles = np.zeros(contour.shape[0])

    # determine the angle between the vectors connecting adjacent points.
    for idx in range(contour.shape[0]):

        point_0 = contour[idx-1]
        point_1 = contour[idx]
        point_2 = contour[idx-2]

        vector_0 = point_0 - point_1
        vector_1 = point_0 - point_2

        norm_0 = np.linalg.norm(vector_0)
        norm_1 = np.linalg.norm(vector_1)

        inner_prod = np.squeeze(np.dot(vector_0, vector_1.T))

        # numerical errors in the inner product can occur if the angle is near
        # 180deg. n.b. due to contour  ordering, angle can't be near zero so
        # cos_angle will never be near 1.
        cos_angle = inner_prod/(norm_0*norm_1)
        if cos_angle < -1:
            cos_angle = -1

        angles[idx] = np.arccos(cos_angle)*180/np.pi

    # veto any angles which are greater than 130 degrees
    corner_idx = np.argwhere(angles < 130)-1

    # keep only 4 of them.
    corner_coords = np.squeeze(contour[corner_idx])[0:4].astype(np.float32)

    warping_coords = np.float32([[500, 10], [10, 10], [10, 500], [500, 500]])

    perspective_matrix = cv2.getPerspectiveTransform(
        corner_coords, warping_coords)

    warped_img = cv2.warpPerspective(img, perspective_matrix, (510, 510))

    return warped_img


def main():
    '''
    demo of the grid extraction
    '''

    img = cv2.imread('crossword1.jpeg')

    cw_contour = find_crossword_contour(img)
    cw_cropped = crop_to_crossword(img, cw_contour)

    cv2.imshow('Crossword', cw_cropped)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
