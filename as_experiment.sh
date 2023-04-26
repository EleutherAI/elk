#!/bin/bash
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --net ccs --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --net ccs --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to10 --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to10 --net ccs --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to50 --num_gpus
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to50 --net ccs --num_gpus
