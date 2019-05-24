#!/bin/env bash

docker
PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py\
    --config=./configs/config.yml\
    --paths=./configs/path.yml\
    --fold=0
