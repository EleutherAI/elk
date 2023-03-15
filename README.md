## Introduction

**WIP: This codebase is under active development**

Because language models are trained to predict the next token in naturally occurring text, they often reproduce common human errors and misconceptions, even when they "know better" in some sense. More worryingly, when models are trained to generate text that's rated highly by humans, they may learn to output false statements that human evaluators can't detect. We aim to circumvent this issue by directly [**eliciting latent knowledge**](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit) (ELK) inside the activations of a language model.

Specifically, we're building on the **Contrast Consistent Search** (CCS) method described in the paper [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) by Burns et al. (2022). In CCS, we search for features in the hidden states of a language model which satisfy certain logical consistency requirements. It turns out that these features are often useful for question-answering and text classification tasks, even though the features are trained without labels.

### Quick **Start**

Our code is based on [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). We test the code on Python 3.9 and 3.10.

First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.

To fit reporters for the HuggingFace model `model` and dataset `dataset`, just run:

```bash
elk elicit microsoft/deberta-v2-xxlarge-mnli imdb
```

This will automatically download the model and dataset, run the model and extract the relevant representations if they aren't cached on disk, fit reporters on them, and save the reporter checkpoints to the `elk-reporters` folder in your home directory. It will also evaluate the reporter classification performance on a held out test set and save it to a CSV file in the same folder.

```bash
elk elicit microsoft/deberta-v2-xxlarge-mnli imdb --net ccs
```
This will generate a CCS reporter instead, of the Eigen reporter, which is the default


## Caching

The hidden states resulting from `elk elicit` are cached as a HuggingFace dataset to avoid having to recompute them every time we want to train a probe. The cache is stored in the same place as all other HuggingFace datasets, which is usually `~/.cache/huggingface/datasets`.

## Other commands

To only extract the hidden states for the model `model` and the dataset `dataset` and save them to `my_output_dir`, without training any reporters, you can run:

```bash
elk extract microsoft/deberta-v2-xxlarge-mnli imdb -o my_output_dir
```

## Development
Use `pip install pre-commit && pre-commit install` in the root folder before your first commit.

### Run tests
```bash
pytest
```
### Run type checking
We use [pyright](https://github.com/microsoft/pyright), which is built into the VSCode editor. If you'd like to run it as a standalone tool, it requires a [nodejs installation.](https://nodejs.org/en/download/)
```bash
pyright
```

If you work on a new feature / fix or some other code task, make sure to create an issue and assign it to yourself (Maybe, even share it in the elk channel of Eleuther's Discord with a small note). In this way, others know you are working on the issue and people won't do the same thing twice 👍 Also others can contact you easily.
