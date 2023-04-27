#!/bin/bash
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_5 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_5 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_10 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_10 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_20 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_20 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_50 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_50 --net ccs --num_gpus 1