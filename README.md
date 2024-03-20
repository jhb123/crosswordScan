
# crosswordScan

crosswordScan is a python tool for extracting a crossword from an image. This is a very early version of the tool and not all of the core functionality is implemented yet. 

Brought to you by the Crossword Scan Gang

## Pipeline

![pipeline](https://github.com/jhb123/crosswordScan/blob/main/crossword_extraction.svg)

## Core Functionality

- [x] https://github.com/jhb123/crosswordScan/issues/1 extraction of the crossword grid.
- [x] https://github.com/jhb123/crosswordScan/issues/2 extraction of the clues.
- [ ] https://github.com/jhb123/crosswordScan/issues/3 interface for completeing the grid.
- [ ] https://github.com/jhb123/crosswordScan/issues/4 connectivity to online database of solutions.

## Other Tasks

- [ ] implement unit testing.
- [ ] aquire larger database of test images.
- [ ] implement CI

## getting crosswordScan

Pull this repo, cd into the directory containing the scr directory and run:
```pip install .```
Try this out in a virtual python environment! import it with `import cws`

## Development

Option 1. Create an environment with docker
```
docker build . --tag experimental_cw_scan
docker run -it experimental_cw_scan /bin/bash
```
Option 2. 
Install OpenCV and Tesseract yourself.

Try out some of the demos functions
```python
from cws import grid_extract
grid_extract.main()
```