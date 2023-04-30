#!/bin/bash
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to10 --num_gpus 1