#!/bin/bash -l

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=elk_imdb_mispellings-rerun

# Remove one # to uncommment
#SBATCH --output=elk_imdb_mispellings-rerun%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0-24:00:00    
#SBATCH --gres=gpu:8

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=sethapun@cs.princeton.edu


# Define and create a unique scratch directory for this job
tag=elk-elk_imdb_mispellings-rerun;
OUT_DIRECTORY='output/'${tag}
mkdir ${OUT_DIRECTORY};

# Submit jobs.
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_10 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_10 --net ccs 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_20 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_20 --net ccs 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_30 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_30 --net ccs 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_40 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_40 --net ccs 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_50 
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_50 --net ccs 

# Finish the script
exit 0
