#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:25:42 2022

@author: josephbriggs
"""
import cws.grid_extract 
import cws.clue_extract
import pytesseract
import importlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract'

    with importlib.resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))
    input_image_copy = input_image.copy()
    grid = cws.grid_extract.digitse_crossword(input_image)
    # clue_marks = cws.grid_extract.get_grid_with_clue_marks(grid)
    
    across_info, down_info = cws.grid_extract.get_clue_info(grid)
    

    all_clues,all_word_lengths,all_clue_lengths = cws.clue_extract.text_box_extraction_pipeline(input_image)
    
    acrosses,downs = cws.clue_extract.match_clues_to_grid(across_info[1],down_info[1],
                                        all_clues,all_word_lengths, all_clue_lengths)
    
   
    fig,ax = plt.subplots()
    ax.imshow(grid)

    print(f'{"  Result  ":#^80}')
    print(f'{"ACROSS":_^80}')
    for n,s,l in zip(across_info[0],*acrosses):
        print(f"{n}a. {s.strip()} {l}")
    print(f'{"DOWN":_^80}')
    for n,s,l in zip(down_info[0],*downs):
        print(f"{n}d. {s.strip()} {l}")
        
    cv2.imshow("Input photo",input_image_copy)
    cv2.waitKey()
    
if __name__ == "__main__":
    main()
