#!/bin/bash

# run in background
nohup python -m elk.extraction_main deberta-v2-xxlarge-mnli super_glue boolq --label-column label --device cuda --max-examples 1000 &
ps -ax | grep extraction_main