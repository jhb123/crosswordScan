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

def test_grid():
    
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
    
    grid = cws.grid_extract.digitse_crossword(input_image)
    assert(np.array_equal(grid, test_grid))
