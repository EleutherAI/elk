#!/bin/bash
elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --num_gpus 1
elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --num_gpus 1
elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to10 --num_gpus 1

elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to50 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to100 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to500 --num_gpus 1

elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/imdb_misspelled_10 --num_gpus 1
elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/imdb_misspelled_30 --num_gpus 1
elk eval arith-huggyllama/llama-13b/sethapun/imdb_misspelled_0/vinc huggyllama/llama-13b sethapun/imdb_misspelled_50 --num_gpus 1

elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/imdb_misspelled_0 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/imdb_misspelled_10 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/imdb_misspelled_30 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/imdb_misspelled_50 --num_gpus 1

elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to10/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --num_gpus 1