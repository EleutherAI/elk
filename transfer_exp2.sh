#!/bin/bash
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to250 --num_gpus 1
elk eval test-huggyllama/llama-13b/sethapun/arithmetic_2as_1to1000/vinc huggyllama/llama-13b sethapun/arithmetic_2as_1to750 --num_gpus 1
