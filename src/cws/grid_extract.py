#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:03:24 2022

@author: josephbriggs
"""
import importlib.resources
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gs_img, (7, 7), 1)
    edges = cv2.Canny(blur, 50, 200, False)
    dilate = cv2.dilate(edges, np.ones((5, 5)), 1)

    # cv2.imshow("edges",edges)
    # cv2.waitKey()

    # find the area of each contour
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.zeros(len(contours))

    for i, contour in enumerate(contours):
        areas[i] = cv2.contourArea(contour)

    # assume image takes up 1/4 of screen
    idxs = np.where(areas > gs_img.size/16)[0]

    # show_contours(img,contours,idxs)

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

    hull = cv2.convexHull(contour)
    epsilon = 0.05*cv2.arcLength(hull, True)

    approx = cv2.approxPolyDP(hull, epsilon, True)

    # cv2.drawContours(img, [contour], -1, (0, 0, 255),  2)
    # cv2.drawContours(img, [hull], -1, (0, 255, 0),  2)
    # cv2.drawContours(img, approx, -1, (255, 0, 0),  10)

    approx_info = np.squeeze(approx)

    corner_coords = approx_info.astype(np.float32)
    warping_coords = np.float32([[500, 10],
                                 [500, 500],
                                 [10, 500],
                                 [10, 10]])

    perspective_matrix = cv2.getPerspectiveTransform(
        corner_coords, warping_coords)

    warped_img = cv2.warpPerspective(img, perspective_matrix, (510, 510))

    # cv2.imshow("cropped", warped_img)
    # cv2.waitKey()

    return warped_img


def get_clue_box_mask(img):
    '''
    Creates a binary image where white is a space for a letter and
    black is a blocked space.

    Parameters
    ----------
    img : np.array
        cropped crossword.

    Returns
    -------
    clue_boxes : np.array

    '''
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gs_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 101, 0)

    kernel = np.ones((9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)


def get_box_size(img):
    '''
    Determine the size of the squares in the crossword

    Parameters
    ----------
    img : np.array
        an image of a crossword.

    Returns
    -------
    float
        The median squaresize found in the crossword grid.

    '''
    # pre-process image
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gs_img, (3, 3), 0)

    # use 2nd derivative to find centre of lines
    grad_x = cv2.Sobel(blur, -1, 2, 0, ksize=3)
    grad_y = cv2.Sobel(blur, -1, 0, 2, ksize=3)

    edges = grad_x+grad_y

    # thresholding improves the performance
    _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # use this to remove small bits of noise.
    kernel = np.ones((3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # locate the squares
    contours, _ = cv2.findContours(
        opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # show_all_contours(img,contours)

    side_length = []

    for contour in contours:

        if ~cv2.isContourConvex(contour):
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if not height == 0:
                aspect = width/height
                if np.abs(1-aspect) < 0.2:
                    side_length.append(width)

    # assume that the most common square is a box for writing an answer in
    return np.median(side_length)


def digitse_crossword(img):
    '''
    convert an image of a crossword into a numpy array.

    Parameters
    ----------
    img : np.array
        image of a crossword.

    Returns
    -------
    np.array

    Each cell in the returned numpy array corresponds to a square in the
    crossword. If the cell == 1 then that is a space for a letter. If the
    cell == 0 then that is a flled in space.
    '''

    cw_contour = find_crossword_contour(img)
    # could replace with match shapes approach?
    cw_cropped = crop_to_crossword(img, cw_contour)[10:500, 10:500]

    box_size = get_box_size(cw_cropped)

    clue_boxes = get_clue_box_mask(cw_cropped)

    # cv2.imshow("clue boxes",clue_boxes)
    # cv2.waitKey()

    rows, cols = clue_boxes.shape

    rows_d = int((rows/box_size))
    cols_d = int((cols/box_size))

    resized_down = cv2.resize(
        clue_boxes, (rows_d, cols_d), interpolation=cv2.INTER_LINEAR)

    resized_down[resized_down == 255] = 1

    return resized_down


def main():
    '''
    example of functions in grid extract
    '''

    test_image = "crossword1.jpeg"
    crossword_location = "cws.resources.crosswords"

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    grid = digitse_crossword(input_image)
    fig, ax = plt.subplots()
    ax.imshow(grid)


if __name__ == "__main__":
    main()
