#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=elk_elicit_tasks_arithmetic

# Remove one # to uncommment
#SBATCH --output=elk_elicit_tasks_arithmetic%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-16:00:00    # Run for 6 hours
#SBATCH --gres=gpu:8

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=sethapun@cs.princeton.edu


# Define and create a unique scratch directory for this job
tag=elk-arithmetic;
OUT_DIRECTORY='output/'${tag}
mkdir ${OUT_DIRECTORY};

# Submit jobs.
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0
elk elicit huggyllama/llama-13b sethapun/imdb_misspelled_0 --net ccs

# Finish the script
exit 0