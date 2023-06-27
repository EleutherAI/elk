#!/bin/bash

csv_file="commands_status.csv"
echo "command,status" > $csv_file

commands=("elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=-1 --net=ccs --norm=burns  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=-1 --net=ccs --norm=leace  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=-1 --net=vinc  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=-1 --net=vinc  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=1 --net=ccs --norm=burns  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=1 --net=ccs --norm=leace  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=1 --net=vinc  --num_gpus 6" "elk sweep --models huggyllama/llama-13b EleutherAI/pythia-12b bigscience/bloom-7b1 --datasets 'ag_news' 'amazon_polarity' 'dbpedia_14' 'glue:qnli' 'imdb' 'piqa' 'super_glue:boolq' 'super_glue:copa' 'super_glue:rte' --binarize --num_variants=1 --net=vinc  --num_gpus 6" )


for command in "${commands[@]}"; do
    echo "$command,NOT STARTED" >> $csv_file
done

for command in "${commands[@]}"; do
    sed -i "s|NOT STARTED|RUNNING|g" $csv_file
    echo "Running command: $command"
    curl -d "Running command: $command" ntfy.sh/derpy
    if ! eval "$command"; then
        sed -i "s|RUNNING|ERROR|g" $csv_file
        echo "Error occurred: Failed to execute command: $command"
        curl -d "Error occurred: Failed to execute command: $command" ntfy.sh/derpy
        break
    else
        sed -i "s|RUNNING|DONE|g" $csv_file
        echo "Command completed successfully: $command"
        curl -d "Command completed successfully: $command" ntfy.sh/derpy
    fi
done
echo 'All combinations completed.'
