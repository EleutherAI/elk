#!/bin/bash

nohup python -u generation_main.py --model deberta-v2-xxlarge-mnli --datasets imdb --prefix confusion --model_device cuda --num_data 1000 &
ps -ax | grep generation_main.py