import base64
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from cws.api import templates
from cws.api.clue import router as clue_router
from cws.api.grid import router as grid_router
from cws import api

logger = logging.getLogger("uvicorn")

app = FastAPI()


templates = Jinja2Templates(directory=templates.__path__)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )
    
app.include_router(clue_router)
app.include_router(grid_router)

api_dir = Path(__file__).parent

app.mount("/static", StaticFiles(directory=api_dir / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)