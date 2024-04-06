#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:19:38 2022

@author: josephbriggs
"""

import importlib.resources
import cv2
import numpy as np
import cws.clue_extract


def test_clue_length_parsing_on_photo_input():

    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))
    grid = cws.grid_extract.digitise_crossword(input_image)
    # clue_marks = cws.grid_extract.get_grid_with_clue_marks(grid)

    across_info, down_info = cws.grid_extract.get_clue_info(grid)

    all_clues, all_word_lengths, all_clue_lengths = cws.clue_extract.text_box_extraction_pipeline(
        input_image)

    acrosses, downs = cws.clue_extract.match_clues_to_grid(across_info[1],
                                                           down_info[1],
                                                           all_clues,
                                                           all_word_lengths,
                                                           all_clue_lengths)

    across_word_length = acrosses[1]
    down_word_length = downs[1]

    expected_across_length = [[6], [8], [9], [5], [4, 7], [3], [7],
                              [2, 1, 3], [6], [7], [3], [11], [5], [9], [8], [6]]

    expected_down_length = [[8], [5], [3], [7], [5, 6], [4, 5], [6], [6],
                            [4, 7], [9], [8], [7], [6], [4, 2], [5], [3]]

    assert(across_word_length == expected_across_length)
    assert(down_word_length == expected_down_length)
