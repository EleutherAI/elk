#!/bin/bash

# run in background
nohup python -m elk.extraction_main deberta-v2-xxlarge-mnli imdb --prefix normal --device cuda --max-examples 1000 &
ps -ax | grep extraction_main
