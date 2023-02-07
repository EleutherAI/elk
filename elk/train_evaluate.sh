#!/bin/bash

# python -m elk.train --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num_data 1000
python -m elk.train --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num_data 1000
python -m elk.evaluate --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num_data 1000
