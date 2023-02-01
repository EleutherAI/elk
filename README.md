### Introduction

**WIP: The code is still actively improved...**
This code is a simplified and refacotored version of the original code of the paper *Discovering Latent Knowledge* in the zip file linked in https://github.com/collin-burns/discovering_latent_knowledge

This code consists of two parts. First, you can use `generation_main.py` to calculate the zero-shot accuracy and generate hidden states. Second, you can use `extraction_main.py` to run our methods, `CCS`, `TPC` and `BSS`, to get final predictions and performance.

We are planning to expand on the method presented in the DLK paper. See our following post, for more information on our plans: https://www.lesswrong.com/posts/bFwigCDMC5ishLz7X/rfc-possible-ways-to-expand-on-discovering-latent-knowledge

### Dependencies

See requirements file

Our code uses [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). You will also need to install [promptsouce](https://github.com/bigscience-workshop/promptsource), a toolkit for NLP prompts. We tested our code on Python 3.8.


### Quick **Start**

For a quick test: You can look into and run generate.sh and evaluate.sh

```bash
sh generate.sh
sh train_evaluate.sh
```

Furthermore:

1. To generate the hidden states for one model `mdl` and all datasets, run

```bash
python generation_main.py --model deberta-v2-xxlarge-mnli --datasets imdb --prefix normal --model_device cuda --num_data 1000
```

To test `deberta-v2-xxlarge-mnli` with the misleading prefix, and only the `imdb` and `amazon-polarity` datasets, while printing extra information, run:

The name of prefix can be found in `./utils_generation/construct_prompts.py`. This command will save hidden states to `generation_results` and will save zero-shot accuracy to `generation_results/generation_results.csv`.

1. To train a ccs model and a logistic regression model

```bash
python train.py --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb  --num_data 1000
```

and evaluate:
```bash
python evaluate.py --dataset imdb
```

Once finished, results will be saved in `evaluation_results/{model}_{prefix}_{seed}.csv`


