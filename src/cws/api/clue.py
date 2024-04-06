import base64
import logging
import tempfile

import cv2
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, File, Request, UploadFile

from cws import grid_extract
from cws.api import templates

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/clue")


templates = Jinja2Templates(directory=templates.__path__)

@router.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="clue_extract.html", context={}
    )
    
    
@router.post("/")
async def upload(request: Request, image: UploadFile = File(...)):
    image = await image.read()
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write the contents of the image into the temporary file
        temp_file.write(image)
        temp_file.flush()
    
        input_image = cv2.imread(temp_file.name)
    
    
    grid = grid_extract.digitise_crossword(input_image)
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
