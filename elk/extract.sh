#!/bin/bash

# run in background
nohup python -m elk.extraction_main --model deberta-v2-xxlarge-mnli --datasets imdb --prefix normal --device cuda --max_examples 1000 &
ps -ax | grep extraction_main
