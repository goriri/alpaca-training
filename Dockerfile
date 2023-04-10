FROM python:3.10-slim

RUN apt update && apt install -y python3-dev gcc

RUN pip3 install sagemaker-pytorch-training

# Copies the training code inside the container
COPY stanford_alpaca /data/sm-alpaca/stanford_alpaca
COPY transformers-zphang /data/sm-alpaca/transformers-zphan

RUN pip3 install -r /data/sm-alpaca/stanford_alpaca/requirements.txt
WORKDIR /data/sm-alpaca/transformers-zphan
RUN pip3 install -e .
