from importlib import resources

import cv2
from cws import grid_extract

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    test_image = "crossword4.jpeg"
    crossword_location = "cws.resources.crosswords"

    with resources.path(crossword_location, test_image) as path:
        input_image = cv2.imread(str(path))

    grid = grid_extract.digitse_crossword(input_image)
    clue_marks = grid_extract.get_grid_with_clue_marks(grid)

    return clue_marks.tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)