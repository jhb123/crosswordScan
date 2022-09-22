#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:40:01 2022

@author: josephbriggs
"""

import string
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def main():
    '''
    tool for generating images of letters from fonts
    '''

    path_to_font = r'/System/Library/Fonts/Supplemental/Arial.ttf'
    # specified font size
    font = ImageFont.truetype(path_to_font, 20)

    letters = string.ascii_letters+string.punctuation+'1234567890'

    for letter in letters:
        image = Image.new("1", (20, 25), 0)

        # drawing text size
        draw = ImageDraw.Draw(image)

        draw.text((2, 0), letter, font=font, fill="white", align="center")

        np_image = np.array(image, dtype=int)
        filled_idx = np.argwhere(np_image == 1)

        left = filled_idx[:, 0].min()
        right = filled_idx[:, 0].max()
        bottom = filled_idx[:, 1].min()
        top = filled_idx[:, 1].max()

        fig, ax = plt.subplots()
        ax.imshow(np_image[left:right+1, bottom:top+1])


if __name__ == "__main__":
    main()
