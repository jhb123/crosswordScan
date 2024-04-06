#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:03:24 2022

@author: josephbriggs
"""
from importlib import resources
import cv2
from matplotlib import pyplot as plt
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


def digitise_crossword(img):
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
    cell == 0 then that is a filled-in space.
    '''

    # cw_contour = find_crossword_contour(img)
    cw_contour = get_grid_contour_by_blobbing(img)
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


def convolve_grid(grid, kernel):
    '''
    wrapper function for filter2d convolve which pads the input
    before performing the convolution and crops the result back
    to the original size.

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.
    kernel : np,array
        a kernel which is convolved with the grid.

    Returns
    -------
    TYPE
        np.array.

    '''
    grid_pad = np.zeros((grid.shape[0]+2, grid.shape[1]+2))
    grid_pad[1:-1, 1:-1] = grid
    tentative = cv2.filter2D(grid_pad, -1, kernel) == 1
    starts = np.logical_and(tentative, grid_pad)
    return starts[1:-1, 1:-1]


def get_grid_with_clue_marks(grid):
    '''
    returns a 2d array with points of interest marked.

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.

    Returns
    -------
    all_info : np.array
        np.array of the same shape as the input grid. if grid[i,j] is:
            0 -> a black square
            1 -> a white square that is not the start of a word
            2 -> the beginning of just an across clue
            3 -> the beginning of just a down clue
            4 -> the beginning of both a down and across clue
    '''
    acc_kernel = np.array([[0, 0, 0],
                          [-1, 0, 1],
                          [0, 0, 0]])

    down_kernel = np.array([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]])

    acc_starts = convolve_grid(grid, acc_kernel)
    down_starts = convolve_grid(grid, down_kernel)
    all_info = grid + acc_starts+2*down_starts
    return all_info


def get_clue_numbers(grid):
    '''
    Returns the numbers and locations of the across and down clues.

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.

    Returns
    -------
        list of the numbers for the across clues
        list of the numbers for the down clues
        list of the start coordinates for the across clues
        list of the start coordinates for the down clues

    '''
    clue_grid = get_grid_with_clue_marks(grid)
    row, col = clue_grid.shape
    acrosses = []
    downs = []
    a_coords = []
    d_coords = []

    idx = 1
    clue_grid = clue_grid.flatten()
    for i, val in enumerate(clue_grid):
        if val == 2:
            acrosses.append(idx)
            a_coords.append([int(i/row), i % col])
            idx = idx + 1
        elif val == 3:
            downs.append(idx)
            d_coords.append([int(i/row), i % col])
            idx = idx+1
        elif val == 4:
            acrosses.append(idx)
            downs.append(idx)
            a_coords.append([int(i/row), i % col])
            d_coords.append([int(i/row), i % col])
            idx = idx + 1
    return acrosses, downs, a_coords, d_coords


def get_across_clue_length(grid, coords):
    '''

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.
    coords : list of np.arrays
        the coordinates of the [row,col] of the start of each word.

    Returns
    -------
        list of lengths ordered in the same way as the coordinates

    '''
    kernel = np.array([[0, 0, 0],
                       [1, 0, 1],
                       [0, 0, 0]])
    bounds = np.logical_and(grid, convolve_grid(grid, kernel))
    bounds_flat = bounds.flatten()

    fill_val = False
    for i, val in enumerate(bounds_flat):
        if val:
            fill_val = ~fill_val
        bounds_flat[i] = fill_val

    across_only = np.reshape(bounds_flat, bounds.shape)
    across_only = np.logical_or(across_only, bounds)

    _, labels = cv2.connectedComponents(
        across_only.astype(np.uint8), 4)

    # fig,ax = plt.subplots()
    # ax.imshow(labels)
    lengths = []
    for i in coords:
        label_num = labels[i[0], i[1]]
        lengths.append(np.count_nonzero(labels == label_num))
    return lengths


def get_down_clue_lengths(grid, coords):
    '''

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.
    coords : list of np.arrays
        the coordinates of the [row,col] of the start of each word.

    Returns
    -------
        list of lengths ordered in the same way as the coordinates

    '''
    kernel = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
    bounds = np.logical_and(grid, convolve_grid(grid, kernel))
    bounds = np.rot90(bounds, 1)

    bounds_flat = bounds.flatten()

    fill_val = False
    for i, val in enumerate(bounds_flat):
        if val:
            fill_val = ~fill_val
        bounds_flat[i] = fill_val

    down_only = np.reshape(bounds_flat, bounds.shape)
    down_only = np.logical_or(down_only, bounds)
    down_only = np.swapaxes(down_only, 0, 1)
    down_only = down_only[:, ::-1]

    _, labels = cv2.connectedComponents(down_only.astype(np.uint8), 4)

    # fig,ax = plt.subplots()
    # ax.imshow(labels)
    lengths = []
    for i in coords:
        label_num = labels[i[0], i[1]]
        lengths.append(np.count_nonzero(labels == label_num))
    return lengths


def get_clue_info(grid):
    '''
    Wrapper for getting clue labels and lengths.

    Parameters
    ----------
    grid : np.array
        a digitised version of a crossword grid.

    Returns
    -------
        tuple of across clue: number, length, start letter location
        tuple of down clue: number, length, start letter location

    '''
    acrosses, downs, a_coords, d_coords = get_clue_numbers(grid)

    across_lengths = get_across_clue_length(grid, a_coords)
    down_lengths = get_down_clue_lengths(grid, d_coords)

    across_info = (acrosses, across_lengths, a_coords)
    down_info = (downs, down_lengths, d_coords)
    return across_info, down_info


def get_grid_contour_by_blobbing(img):
    '''
    uses connected component analysis to find the crossword.

    Parameters
    ----------
    img : np.array
        3 channel image containing a crossword.

    Returns
    -------
    TYPE
        contour that encloses the crossword.

    '''

    #pre-process
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gs_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 30)

    dilate = cv2.dilate(thresh, np.ones((7, 7)), 1)


    _, label_ids, values, _ = cv2.connectedComponentsWithStats(dilate, 8)

    areas = values[:, 4]

    # assumes that the crossword takes up 1/6 of the page.
    area_thresh = gs_img.size/(6**2)
    idxs = np.argwhere(areas > area_thresh)

    big_features = values[idxs, :]
    widths = np.squeeze(big_features)[:, 2]
    heights = np.squeeze(big_features)[:, 3]

    aspects = widths/heights

    # assume that the crossword is the squarest big blob.
    squareness = np.abs(aspects - 1)
    squarest_blob_i = np.argmin(squareness)

    # use the info from big_features to search the original blob results.
    row_info = np.squeeze(big_features[squarest_blob_i, :])
    row = np.argwhere(values[:, :] == row_info)
    cw_label = row[0][0]

    crossword_blob = label_ids == cw_label
    crossword_blob = 255*crossword_blob.astype(np.uint8)

    contours, _ = cv2.findContours(crossword_blob,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]


def main():
    '''
    example of functions in grid extract
    '''

    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"

    with resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    grid = digitise_crossword(input_image)
    clue_marks = get_grid_with_clue_marks(grid)
    across_info, down_info = get_clue_info(grid)

    a_string_info = [f' {c[0]}a. ({c[1]}) at {c[2]}'
                     for c in zip(across_info[0], across_info[1], across_info[2])]
    d_string_info = [f' {c[0]}a. ({c[1]}) at {c[2]}'
                     for c in zip(down_info[0], down_info[1], down_info[2])]

    print(*a_string_info, sep='\n')
    print('\n')
    print(*d_string_info, sep='\n')

    _, ax = plt.subplots()
    ax.imshow(clue_marks)
    plt.show()

if __name__ == "__main__":
    main()
