#!/bin/bash
elk eval test-huggyllama/llama-13b/sethapun/imdb_misspelled_0/ccs huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/imdb_misspelled_0/ccs huggyllama/llama-13b sethapun/arithmetic_2as_1to50 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/imdb_misspelled_0/ccs huggyllama/llama-13b sethapun/arithmetic_2as_1to250 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/imdb_misspelled_0/ccs huggyllama/llama-13b sethapun/arithmetic_2as_1to500 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/imdb_misspelled_0/ccs huggyllama/llama-13b sethapun/arithmetic_2as_1to750 --num_gpus 1