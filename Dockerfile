FROM python:3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tesseract-ocr  -y

WORKDIR /app

ENV TESSERACT_PATH=/usr/bin/tesseract

COPY . .

RUN pip install .

CMD [ "pytest" ]