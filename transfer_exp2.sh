#!/bin/bash

elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/ccs huggyllama/llama-13b sethapun/imdb_misspelled_0 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/ccs huggyllama/llama-13b sethapun/imdb_misspelled_10 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/ccs huggyllama/llama-13b sethapun/imdb_misspelled_30 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/ccs huggyllama/llama-13b sethapun/imdb_misspelled_50 --num_gpus 1
