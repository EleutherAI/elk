#!/bin/bash

python -m elk.train suspicious-wu deberta-v2-xxlarge-mnli imdb
# python -m elk.evaluate deberta-v2-xxlarge-mnli boolq --prefix normal --num-data 1000
