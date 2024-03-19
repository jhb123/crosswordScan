#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:05:52 2022

@author: josephbriggs
"""
import importlib
import random as rng
import string
import re
import cv2
import numpy as np
import pytesseract
import cws.grid_extract


def match_template(arr, pattern):
    '''
    wrapper function for cv2.matchTemplate
    '''
    matched_template = cv2.matchTemplate(arr.astype(
        'uint8'), pattern.astype('uint8'), cv2.TM_SQDIFF)
    matched_template = matched_template.ravel()
    return matched_template


def segment_page_preprocess(img):
    '''
    locates text in an image and returns a binary mask.

    Parameters
    ----------
    img : np.array
        3 channel image to extract clues from.

    Returns
    -------
    np.array
        1 channel, binary image. White corresponds to the location of
        letters

    '''

    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_for_reading = cv2.adaptiveThreshold(gs_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 9, 15)

    # img_for_reading = cv2.erode(img_for_reading,np.ones((3,3)),1)
    whitelist = string.ascii_letters + '1234567890'

    configs = ["--psm 3",
               "-c tessedit_char_whitelist={whitelist}"]
    cfg_str = " ".join(configs)

    # also include any config options you use
    data = pytesseract.image_to_data(img_for_reading, config=cfg_str)
    lefts = [re.split('\t', d)[6] for d in data.splitlines()[1:]]
    tops = [re.split('\t', d)[7] for d in data.splitlines()[1:]]
    widths = [re.split('\t', d)[8] for d in data.splitlines()[1:]]
    heights = [re.split('\t', d)[9] for d in data.splitlines()[1:]]
    texts = [re.split('\t', d)[11] for d in data.splitlines()[1:]]

    word_loc = np.zeros(img_for_reading.shape, dtype=np.uint8)

    lefts = np.array([int(i) for i in lefts])
    tops = np.array([int(i) for i in tops])
    widths = np.array([int(i) for i in widths])
    heights = np.array([int(i) for i in heights])
    # confs = np.array([float(i) for i in confs])

    heights_for_average = []

    whitelist = rf'[\d{string.ascii_letters}]'

    for left, top, width, height, text in zip(lefts, tops, widths, heights, texts):

        is_ok = bool(re.search(whitelist, text))
        if is_ok:
            heights_for_average.append(height)
            right = left + width
            # tesseract and opencv have different definitions of top and bottom
            bottom = height + top

            cv2.rectangle(word_loc,
                          (left, top),
                          (right, bottom),
                          255, -1)

    return word_loc, np.median(heights_for_average)


def segment_page_idxs(word_loc, smoothing=100, thresh_f=5):
    '''

    provides indexes for x text boxes based on white space
    analysis.

    Parameters
    ----------
    word_loc : np.array
        1 channel image showing the location of text (black == text)
    smoothing : int, optional
       bigger number, less smoothing. the segmmenter is very sensitive to
       this parameter. The default is 100.
    thresh_f : int, optional
        percentage of row that is white that will be counted as a
        empty row. The default is 5.

    Returns
    -------
    np.arary of row indexes which are white.

    '''
    row_scan = np.sum(word_loc, axis=1)

    scan_smooth_factor = smoothing
    size = int(row_scan.size/scan_smooth_factor)
    if size % 2 == 0:
        corr = 1
    else:
        corr = 0
    smoother = np.ones(size + corr)/(size+corr)
    row_scan_s = np.convolve(row_scan, smoother, mode='valid')

    thresh = np.percentile(row_scan_s, [thresh_f])

    white_spaces = np.argwhere(row_scan_s <= thresh)

    return white_spaces


def segment_page(word_loc):
    '''
    provides indexes for x and y positions of text boxes based on white space
    analysis. wraps segment_page_idxs. flips the image to find the y indexes.

    Parameters
    ----------
    word_loc : np.array
        1 channel image showing the location of text (black == text)

    Returns
    -------
    x_idxs : np.array
        indexes of empty columns.
    y_idxs : np.array
        indexes of empty rows.

    '''
    x_idxs = segment_page_idxs(word_loc,
                               smoothing=100, thresh_f=10)
    y_idxs = segment_page_idxs(np.swapaxes(word_loc, 0, 1),
                               smoothing=100, thresh_f=10)

    return x_idxs, y_idxs


def get_text_boxes(img):
    '''
    splits an input image into rectangles based on white space and then
    performs connected component analysis. See cv2 documentation for
    explanation of the outputs.

    Parameters
    ----------
    img : np.array
        1 channel thesholded image.

    Returns
    -------
    output of cv2.connectedComponentsWithStats

    '''
    x_idxs, y_idxs = segment_page(img)

    text_boxes_mask = 255*np.ones(img.shape, dtype=np.uint8)

    text_boxes_mask[x_idxs, :] = 0
    text_boxes_mask[:, y_idxs] = 0

    total_labels, labels, values, centroid = cv2.connectedComponentsWithStats(
        text_boxes_mask, 8)

    return total_labels, labels, values, centroid


def get_text_box_idx(img, labels, idx, pad):
    '''
    return a cropped version of the input based on the location of a
    mask of rectangles obtained via blobbing.

    Parameters
    ----------
    img : np.array
        full image.
    labels : np.array
        labels obtained from cv2.connectedComponents... These should be
        rectangular blobs
    idx : int
        the index of the blob that is to be cropped to.
    pad : int
        the amount by which the blob shouuld be expanded by.

    Returns
    -------
        np.array

    '''

    mask = labels == idx
    mask = 255*mask.astype(np.uint8)
    kernel = np.ones((pad, pad))
    mask_dilate = cv2.dilate(mask, kernel, 1)
    mask_array = np.ix_(mask_dilate.any(1), mask_dilate.any(0))
    return img[mask_array]


def text_box_pre_process(img):
    '''
    preprocesses an image for text scanning.

    Parameters
    ----------
    img : np.array
        This is a 3 channel image. It should be cropped to only have text to
        improve the speed of the preprocessing.

    Returns
    -------
    img_for_reading : np.array
        a x4 higher resolution that has been thresholded.

    '''

    super_resolution = cv2.dnn_superres.DnnSuperResImpl_create()

    model_file = "ESPCN_x4.pb"
    model_location = "cws.resources.neural_net_models"
    with importlib.resources.path(model_location, model_file) as path:
        super_resolution.readModel(str(path))

    super_resolution.setModel("espcn", 4)
    result = super_resolution.upsample(img)

    gs_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    img_for_reading = cv2.medianBlur(gs_img, 5)

    img_for_reading = cv2.adaptiveThreshold(img_for_reading,
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            201,
                                            30)

    # kernel = np.ones((5,5))
    # img_for_reading = cv2.morphologyEx(img_for_reading, cv2.MORPH_CLOSE, kernel)
    # blur = cv2.GaussianBlur(img_for_reading,(9,9),1)

    # cv2.imshow("text to analyse",img_for_reading)
    # cv2.waitKey()

    return img_for_reading


def text_box_clue_extraction(img):
    '''
    converts an image of text into lists of clues and meta-info.

    Parameters
    ----------
    img : np.array()
        pre-processed image of containing clues.

    Returns
    -------
     clues : list
         list of strings for each clue. this is obtained by OCR.
     word_lengths : list
         list of a list of ints for corresponding to length of each word within
         a clue e.g. a [4,2] clue would be [4,2] and a 6 clue would be [6]).
         This is obtained by OCR.
     clue_lengths : list
         list of ints for the total length of each clue as extracted by OCR.
    raw_text : string
        the whole tesseract ouput without any editing.

    '''

    # print(pytesseract.image_to_osd(img))
    # try --psm 3
    all_text = pytesseract.image_to_string(img, config='--psm 3')
    raw_text = all_text

    # makes it easier to process the text if there is only one type of bracket.
    left_brackets = r"[{\[]"
    right_brackets = r"[\|}\]]"
    all_text = re.sub(left_brackets, '(', all_text)
    all_text = re.sub(right_brackets, ')', all_text)

    # look for numbers with at least one bracket or no bracket but a new line.
    # some common misclassified numbers are 6-> G and 5->S and sometimes 8->g
    pattern = r'(\(?[\d.\-,gsGS\s]*\))|(\([\d.\-,gsGS\s]*\)?)|(\d+\n)|\s[GSsg]\n'

    split_text = re.split(pattern, all_text)
    # print(split_text)
    split_text = [s for s in split_text if s is not None]
    split_text = [s.replace('\n', ' ') for s in split_text]
    split_text = [s for s in split_text if s != '']
    split_text = [s for s in split_text if s != ' ']

    # if there were any splits, that means there was a clue detected.
    # the split_text_list will follow a [clue_0,clue_0_length,...
    # clue_n,clue_n_length] structure.

    if len(split_text) > 1:

        clues = split_text[::2]
        word_lengths_str = split_text[1::2]

    else:
        clues = []
        word_lengths_str = []
        clue_lengths = []

    # convert a string like '(4,2)' into a list like [4,2].
    word_lengths = [list(map(int, re.findall(r'\d+', s)))
                    for s in word_lengths_str]
    clue_lengths = [sum(l) for l in word_lengths]

    return clues, word_lengths, clue_lengths, raw_text


def show_box_areas(total_labels, labels):
    '''
    debug function that takes the output  of cv2.connectedComponentsWithStats
    and an image. Draws where the boxes are on a blank canvas.
    '''
    debug_boxes = np.zeros(
        (labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for i in range(total_labels):
        colour = (rng.randint(0, 256), rng.randint(
            0, 256), rng.randint(0, 256))
        debug_boxes[labels == i] = colour

    cv2.imshow("text boxes", debug_boxes)
    cv2.waitKey()


def show_box_areas_over_img(total_labels, stats, centre, img):
    '''
    debug function that takes the output  of cv2.connectedComponentsWithStats
    and an image. Draws where the boxes are on the image.
    '''
    img_to_label = np.copy(img)

    for i in range(total_labels):
        colour = (rng.randint(0, 256),
                  rng.randint(0, 256),
                  rng.randint(0, 256))

        left = stats[i, 0]
        right = left+stats[i, 2]
        top = stats[i, 1]
        bottom = right+stats[i, 3]
        # tesseract and opencv have different definitions of top and bottom

        cv2.rectangle(img_to_label,
                      (left, top),
                      (right, bottom),
                      colour, 5)
        cv2.putText(img_to_label,
                    str(i),
                    (int(centre[i][0]), int(centre[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    colour,
                    10,
                    cv2.LINE_4)

    cv2.imshow("text boxes", img_to_label)
    cv2.waitKey()


def text_box_extraction_pipeline(input_image):
    '''
    extracts clues from a photo of a crossword.

    Parameters
    ----------
    input_image : np.array
        image of a page containing crossword clues.

    Returns
    -------
    all_clues : list
        a list of strings corresponding to clues found on the page.
    all_word_lengths : list
        a list of list of ints (not a typo) corresponding to length of words
        within a clue i.e. captures whether a clue has a multiple word answer.
    all_clue_lengths : list
        int corresponding to the total length of a clue. This is obtained via
        OCR, and so it is not 100% reliable. Use cws.grid_extract.get_clue_info
        to obtain an accurate version of this information. all_clue_lengths is
        only required for matching clues to grid location.

    '''

    # remove grid
    cross_word_contour = cws.grid_extract.get_grid_contour_by_blobbing(
        input_image)
    cv2.fillPoly(input_image, [cross_word_contour], [255, 255, 255])

    word_loc, _ = segment_page_preprocess(input_image)

    total_labels, labels, _, _ = get_text_boxes(word_loc)

    all_clues = []
    all_word_lengths = []
    all_clue_lengths = []

    for i in range(1, total_labels):
        cropped_text_box = get_text_box_idx(input_image, labels, i, 1)
        cropped_text_box_pre_processed = text_box_pre_process(
            cropped_text_box)

        clues, word_lengths, clue_lengths, _ = text_box_clue_extraction(
            cropped_text_box_pre_processed)

        all_clues = all_clues+clues
        all_word_lengths = all_word_lengths + word_lengths
        all_clue_lengths = all_clue_lengths + clue_lengths

    return all_clues, all_word_lengths, all_clue_lengths


def match_clues_to_grid(a_clue_length, d_clue_length, all_clues,
                        all_word_lengths, all_clue_lengths):
    '''
    Performs match filtering to try and match clues to their position in the
    grid. Use the outputs of cws.grid_extract and
    cws.clue_extract.text_box_extraction_pipeline as the inputs.

    Parameters
    ----------
    a_clue_length : list
        list of ints corresponding to the number of connected white squares
        for each across clue (e.g. a [4,2] clue would be 6).
        This can be obtained purely from the image of the grid.
    d_clue_length : list
        list of ints corresponding to the number of connected white squares
        for each down clue (e.g. a [4,6] clue would be 6).
        This can be obtained purely from the image of the grid.
    all_clues : list
        list of strings for each clue. this is obtained by OCR.
    all_word_lengths : list
        list of a list of ints for corresponding to length of each word within
        a clue e.g. a [4,2] clue would be [4,2] and a 6 clue would be [6]).
        This is obtained by OCR.
    all_clue_lengths : list
        list of ints for the total length of each clue as extracted by OCR.

    Returns
    -------
    TYPE
        tuple.
        the 0th element of the tuple is a tuple containing:
            * a list of the across clues
            * a list of the across clue lengths
        the 1st element of the tuple is a tuple containing:
            * a list of the down clues
            * a list of the down clue lengths

    '''

    a_match_filter_result = match_template(
        np.array(all_clue_lengths), np.array(a_clue_length))
    d_match_filter_result = match_template(
        np.array(all_clue_lengths), np.array(d_clue_length))

    a_idx_start = np.argmin(a_match_filter_result)
    across_clues = all_clues[a_idx_start:a_idx_start+len(a_clue_length)]
    across_clue_lengths = all_word_lengths[a_idx_start:a_idx_start+len(
        a_clue_length)]

    d_idx_start = np.argmin(d_match_filter_result)
    down_clues = all_clues[d_idx_start:d_idx_start+len(d_clue_length)]
    down_clue_lengths = all_word_lengths[d_idx_start:d_idx_start +
                                         len(d_clue_length)]


    across_clue_lengths = choose_grid_or_ocr_length(a_clue_length,across_clue_lengths)
    down_clue_lengths = choose_grid_or_ocr_length(d_clue_length,down_clue_lengths)


    return (across_clues, across_clue_lengths), (down_clues, down_clue_lengths)

def choose_grid_or_ocr_length(grid_legnth,ocr_length):
    clue_length = []
    for grid_len, ocr_len in zip(grid_legnth,ocr_length):
        if len(ocr_len) > 1:
            clue_length.append(ocr_len)
        else:
            clue_length.append([grid_len])

    return clue_length


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

    all_clues, all_word_lengths, all_clue_lengths = text_box_extraction_pipeline(
        input_image)

    acrosses, downs = match_clues_to_grid(across_info[1], down_info[1],
                                          all_clues, all_word_lengths, all_clue_lengths)

    fig, ax = plt.subplots()
    ax.imshow(grid)

    print(f'{"  Result  ":#^80}')
    print(f'{"ACROSS":_^80}')
    for n, s, l in zip(across_info[0], *acrosses):
        print(f"{n}a. {s.strip()} {l}")
    print(f'{"DOWN":_^80}')
    for n, s, l in zip(down_info[0], *downs):
        print(f"{n}d. {s.strip()} {l}")

    cv2.imshow("Input photo", input_image_copy)
    cv2.waitKey()


if __name__ == "__main__":
    main()
