#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:31:36 2022

@author: josephbriggs
"""

import importlib.resources
import cv2
import numpy as np
import cws.grid_extract
import pytest


def test_grid_digitisation_with_photo():

    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    test_grid = np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                          [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
                          [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                          [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                          [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]])

    grid = cws.grid_extract.digitise_crossword(input_image)
    assert(np.array_equal(grid, test_grid))


def squew_square():

    image = np.zeros((101, 101, 3), dtype=np.uint8)

    image[30:70, 30:70, :] = 255

    pts1 = np.array([[30, 30], [70, 30], [30, 70], [70, 70]]
                    ).astype(np.float32)
    pts2 = np.array([[30, 26], [70, 30], [31, 68], [67, 71]]
                    ).astype(np.float32)

    transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distortion = cv2.warpPerspective(image, transform_matrix, (101, 101))

    return distortion


def central_white_square():

    image = np.zeros((101, 101, 3), dtype=np.uint8)

    image[30:70, 30:70, :] = 255

    return image


@pytest.mark.parametrize('img', [central_white_square(), squew_square()])
def test_get_grid_contour_by_blobbing_(img):
    contour = cws.grid_extract.find_crossword_contour(img)
    contour_area = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(contour_area, [contour], -1, 255, -1)

    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    delta = np.abs(contour_area-gs_img)

    left_over = np.sum(delta)/(255*gs_img.size)
    assert(left_over < 0.05)
