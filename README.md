## Introduction

**WIP: This codebase is under active development**

Because language models are trained to predict the next token in naturally occurring text, they often reproduce common human errors and misconceptions, even when they "know better" in some sense. More worryingly, when models are trained to generate text that's rated highly by humans, they may learn to output false statements that human evaluators can't detect. We aim to circumvent this issue by directly [**eliciting latent knowledge**](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit) (ELK) inside the activations of a language model.

Specifically, we're building on the **Contrast Consistent Search** (CCS) method described in the paper [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) by Burns et al. (2022). In CCS, we search for features in the hidden states of a language model which satisfy certain logical consistency requirements. It turns out that these features are often useful for question-answering and text classification tasks, even though the features are trained without labels.

### Quick **Start**

Our code is based on [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). We test the code on Python 3.9 and 3.10.

First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.


To extract the hidden states for one model `model` and the dataset `dataset` *and* train a probe on these extracted hidden states, run:

```bash
elk elicit microsoft/deberta-v2-xxlarge-mnli imdb
```

To only extract the hidden states for one model `model` and the dataset `dataset`, run:

```bash
elk extract microsoft/deberta-v2-xxlarge-mnli imdb
```

and evaluate on different datasets: [WIP]

### Distributed hidden state extraction

You can run the hidden state extraction in parallel on multiple GPUs with `torchrun`. Specifically, you can run the hidden state extraction using all GPUs on a node with:

```bash
torchrun --nproc_per_node gpu -m elk extract microsoft/deberta-v2-xxlarge-mnli imdb
```

Currently, our code doesn't quite support distributed training of the probe. Running `elk elicit` under `torchrun` tends to hang during the training phase. We're working on fixing this.

## Caching

We cache the hidden states resulting from `elk extract` to avoid having to recompute them every time we want to train a probe. The cache is stored in `~/.cache/elk/{md5_hash_of_cli_args}`. Probes are also cached alongside the hidden states they were trained on. You can see a summary of all the cached hidden states by running `elk list`.

## Development
To clone the repo and its submodules
```bash
git clone --recurse-submodules https://github.com/EleutherAI/elk.git
```


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
