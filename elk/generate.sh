#!/bin/bash

# run in background
nohup python -m elk.generation_main --model deberta-v2-xxlarge-mnli --datasets imdb --prefix normal --device cuda --num-data 1000 &
ps -ax | grep generation_main
