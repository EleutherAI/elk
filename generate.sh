#!/bin/bash

# run in background
nohup python -u generation_main.py --model deberta-v2-xxlarge-mnli --datasets imdb --prefix normal --model_device cuda --num_data 1000 &
ps -ax | grep generation_main.py
