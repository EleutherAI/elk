#!/bin/bash

python train.py --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num_data 1000
python evaluate.py --model deberta-v2-xxlarge-mnli --prefix normal  --dataset boolq  --num_data 1000
