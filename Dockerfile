FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

COPY . .

RUN /app/scripts/install.sh

