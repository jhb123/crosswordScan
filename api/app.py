import base64
from importlib import resources
import logging
import tempfile
from typing import Annotated

import cv2
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from cws import grid_extract

from fastapi import FastAPI, File, Form, Request, UploadFile
import uvicorn

logger = logging.getLogger("uvicorn")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )
    
    
@app.post("/")
async def upload(request: Request, image: UploadFile = File(...)):
    image = await image.read()
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write the contents of the image into the temporary file
        temp_file.write(image)
        temp_file.flush()
    
        input_image = cv2.imread(temp_file.name)
    
    
    grid = grid_extract.digitse_crossword(input_image)
    clue_marks = grid_extract.get_grid_with_clue_marks(grid)
    logger.info(f"\n{grid}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        resized_array = cv2.resize(clue_marks*100, (201, 201), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(temp_file.name, resized_array)

        img_data = temp_file.read()

        img_base64 = base64.b64encode(img_data).decode("utf-8")

        # Construct the Data URI
        data_uri = f"data:image/png;base64,{img_base64}"

        return templates.TemplateResponse(
            request=request, name="embedded_image.html", context={"data_uri": data_uri}
        )

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)