FROM python:3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tesseract-ocr  -y

WORKDIR /app

ENV TESSERACT_PATH=/usr/bin/tesseract

COPY . .

RUN pip install .

WORKDIR /app/api

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]