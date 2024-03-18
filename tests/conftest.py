#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 22:33:58 2022

@author: josephbriggs
"""

import numpy as np
import cv2
import pytest


@pytest.fixture
def central_white_square():

    image = np.zeros((101, 101, 3), dtype=np.uint8)

    image[30:70, 30:70, :] = 255

    return image


@pytest.fixture
def two_squares():

    image = np.zeros((101, 101, 3), dtype=np.uint8)

    image[3:20, 3:20, :] = 255
    image[40:90, 40:90, :] = 255

    return image


@pytest.fixture
def square_and_rectangle():

    image = np.zeros((101, 101, 3), dtype=np.uint8)

    image[3:20, 3:20, :] = 255
    image[40:90, 40:90, :] = 255

    return image


# @pytest.fixture
# def squew_square():

#     image = np.zeros((101, 101, 3), dtype=np.uint8)

#     image[30:70, 30:70, :] = 255

#     pts1 = np.array([[30, 30], [70, 30], [30, 70], [70, 70]]
#                     ).astype(np.float32)
#     pts2 = np.array([[30, 26], [70, 30], [31, 68], [67, 71]]
#                     ).astype(np.float32)

#     transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     distortion = cv2.warpPerspective(image, transform_matrix, (101, 101))

#     return distortion
