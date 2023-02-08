### Introduction

**WIP: This codebase is under active development**
This code is a simplified and refactored version of the original code of the paper *Discovering Latent Knowledge* in the zip file linked in https://github.com/collin-burns/discovering_latent_knowledge

### Dependencies

See requirements.txt file

Our code uses [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). You will also need to install [promptsouce](https://github.com/bigscience-workshop/promptsource), a toolkit for NLP prompts. We tested our code on Python 3.9.


### Quick **Start**

First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.

```bash

# Extract hidden states from the deberta model for prompts generated from the imdb dataset
# By default, this will create a new folder with a randomly generated name, e.g. eager-yonath, under /home/.cache/elk/
# The folder will contain the hidden states and other important files
elk extract deberta-v2-xxlarge-mnli imdb

# Set the above mentioned folder name to train the probe
elk train eager-yonath
```

### Development

Use `pip install pre-commit && pre-commit install` in the root folder before your first commit.

If you work on a new feature / fix or some other code task, make sure to create an issue and assign it to yourself (Maybe, even share it in the elk channel of Eleuther's Discord with a small note). In this way, others know you are working on the issue and people won't do the same thing twice 👍 Also others can contact you easily.
