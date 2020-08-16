FROM gcr.io/kaggle-gpu-images/python:latest

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /srv
COPY . .
RUN pip install -e .[dev]
