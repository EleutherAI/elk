### Introduction

**WIP: This codebase is under active development**
This code is a simplified and refactored version of the original code of the paper *Discovering Latent Knowledge* in the zip file linked in https://github.com/collin-burns/discovering_latent_knowledge

### Dependencies

See requirements.txt file

Our code uses [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). You will also need to install [promptsouce](https://github.com/bigscience-workshop/promptsource), a toolkit for NLP prompts. We tested our code on Python 3.9.


### Quick **Start**

First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.

For a quick test: You can look into and run extract.sh and evaluate.sh (**Warning:** They are in the package elk itself right now. Will be changed):

```bash
cd elk
sh extract.sh
sh train_evaluate.sh
```

Furthermore:

1. To extract the hidden states for one model `mdl` and all datasets, `cd elk` and then run

```bash
python extraction_main.py --model deberta-v2-xxlarge-mnli --datasets imdb --prefix normal --device cuda --num_data 1000
```

To test `deberta-v2-xxlarge-mnli` with the misleading prefix, and only the `imdb` and `amazon-polarity` datasets, while printing extra information, run:

The name of prefix can be found in `./extraction/construct_prompts.py`. This command will save hidden states to `extraction_results` and will save zero-shot accuracy to `extraction_results/extraction_results.csv`.

1. To train a ccs model and a logistic regression model

```bash
python train.py --model deberta-v2-xxlarge-mnli --prefix normal --dataset imdb --num-data 1000
```

and evaluate:
```bash
python evaluate.py --model deberta-v2-xxlarge-mnli --dataset imdb --num-data 1000
```

Once finished, results will be saved in `evaluation_results/{model}_{prefix}_{seed}.csv`

### Development

Use `pip install pre-commit && pre-commit install` in the root folder before your first commit.

If you work on a new feature / fix or some other code task, make sure to create an issue and assign it to yourself (Maybe, even share it in the elk channel of Eleuther's Discord with a small note). In this way, others know you are working on the issue and people won't do the same thing twice 👍 Also others can contact you easily.
