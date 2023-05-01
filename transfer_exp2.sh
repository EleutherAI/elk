#!/bin/bash

elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to50 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to250 --net ccs --num_gpus 1
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to750 --net ccs --num_gpus 1


