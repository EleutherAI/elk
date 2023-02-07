#!/bin/bash

# run in background
nohup python -m elk.generation_main --model deberta-v2-xxlarge-mnli --datasets imdb  &
ps -ax | grep generation_main
