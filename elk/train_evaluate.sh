#!/bin/bash

# python -m elk.train --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num-data 1000
python -m elk.train --model deberta-v2-xxlarge-mnli --prefix normal  --dataset boolq  --num-data 1000
python -m elk.evaluate --model deberta-v2-xxlarge-mnli --prefix normal  --dataset boolq  --num-data 1000
