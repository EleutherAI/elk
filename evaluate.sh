#!/bin/bash

python evaluate.py --model deberta-v2-xxlarge-mnli --prefix confusion  --dataset imdb  --num_data 1000
# python extraction_main.py --model deberta-v2-xxlarge-mnli --prefix confusion  --dataset boolq  --num_data 1000
