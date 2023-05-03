## Introduction

**WIP: This codebase is under active development**

Because language models are trained to predict the next token in naturally occurring text, they often reproduce common human errors and misconceptions, even when they "know better" in some sense. More worryingly, when models are trained to generate text that's rated highly by humans, they may learn to output false statements that human evaluators can't detect. We aim to circumvent this issue by directly [**eliciting latent knowledge**](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit) (ELK) inside the activations of a language model.

Specifically, we're building on the **Contrastive Representation Clustering** (CRC) method described in the paper [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) by Burns et al. (2022). In CRC, we search for features in the hidden states of a language model which satisfy certain logical consistency requirements. It turns out that these features are often useful for question-answering and text classification tasks, even though the features are trained without labels.

### Quick **Start**

Our code is based on [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). We test the code on Python 3.10 and 3.11.

First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.

To fit reporters for the HuggingFace model `model` and dataset `dataset`, just run:

```bash
elk elicit microsoft/deberta-v2-xxlarge-mnli imdb
```

This will automatically download the model and dataset, run the model and extract the relevant representations if they aren't cached on disk, fit reporters on them, and save the reporter checkpoints to the `elk-reporters` folder in your home directory. It will also evaluate the reporter classification performance on a held out test set and save it to a CSV file in the same folder.

The following will generate a CCS (Contrast Consistent Search) reporter instead of the CRC-based reporter, which is the default.

```bash
elk elicit microsoft/deberta-v2-xxlarge-mnli imdb --net ccs
```

The following command will evaluate the probe from the run naughty-northcutt on the hidden states extracted from the model deberta-v2-xxlarge-mnli for the imdb dataset. It will result in an `eval.csv` and `cfg.yaml` file, which are stored under a subfolder in `elk-reporters/naughty-northcutt/transfer_eval`.

```bash
elk eval naughty-northcutt microsoft/deberta-v2-xxlarge-mnli imdb
```

The following runs `elicit` on the Cartesian product of the listed models and datasets, storing it in a special folder ELK_DIR/sweeps/<memorable_name>. Moreover, `--add_pooled` adds an additional dataset that pools all of the datasets together.

```bash
elk sweep --models gpt2-{medium,large,xl} --datasets imdb amazon_polarity --add_pooled
```

## Running big models
For big models that cannot fit on a single gpu, you'll need to use multiple
gpus per model.

This is an example to run a single 8bit llama-65b model on 2 A40s that have
~50 GB of memory each.

```
elk elicit huggyllama/llama-65b imdb --num_gpus 2 --gpus_per_model 2 --int8
```

## Caching

The hidden states resulting from `elk elicit` are cached as a HuggingFace dataset to avoid having to recompute them every time we want to train a probe. The cache is stored in the same place as all other HuggingFace datasets, which is usually `~/.cache/huggingface/datasets`.

## Development
Use `pip install pre-commit && pre-commit install` in the root folder before your first commit.

### Devcontainer

[
    ![Open in Remote - Containers](
        https://img.shields.io/static/v1?label=Remote%20-%20Containers&message=Open&color=blue&logo=visualstudiocode
    )
](
    https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/EleutherAI/elk
)

### Run tests
```bash
pytest
```
### Run type checking
We use [pyright](https://github.com/microsoft/pyright), which is built into the VSCode editor. If you'd like to run it as a standalone tool, it requires a [nodejs installation.](https://nodejs.org/en/download/)
```bash
pyright
```

### Run the linter
We use [ruff](https://beta.ruff.rs/docs/). It is installed as a pre-commit hook, so you don't have to run it manually.
If you want to run it manually, you can do so with:
```bash
ruff . --fix
```

### Contributing to this repository

If you work on a new feature / fix or some other code task, make sure to create an issue and assign it to yourself (Maybe, even share it in the elk channel of Eleuther's Discord with a small note). In this way, others know you are working on the issue and people won't do the same thing twice 👍 Also others can contact you easily.
