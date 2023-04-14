#!/bin/bash
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to2
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to5
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to1 --net ccs
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to2 --net ccs
elk elicit huggyllama/llama-13b sethapun/arithmetic_2as_1to5 --net ccs

