## Introduction

**WARNING**: This code is actively being hacked on and is probably unstable in all sorts of ways. Proceed with caution. If you'd like to help improve the code, please check out the #eliciting-latent-knowledge channel on the EleutherAI Discord server.

This code consists of two parts. First, you can use `generation_main.py` to calculate the zero-shot accuracy and generate hidden states. Second, you can use `extraction_main.py` to run our methods, `CCS`, `TPC` and `BSS`, to get final predictions and performance.


## Quick Start

1. First install the package with `pip install -e .` in the root directory, or `pip install -e .[dev]` if you'd like to contribute to the project (see **Development** section below). This should install all the necessary dependencies.
2. To generate the hidden states for one model `<MODEL>` and all datasets, run

```bash
python generation_main.py --model <MODEL> --swipe
```

To test `roberta` with the misleading prefix, and only the `imdb` and `amazon-polarity` datasets, while printing extra information, run:


```bash
python generation_main.py --model roberta-large-mnli --prefix confusion --swipe --datasets imdb amazon-polarity --print_more
```

The name of prefix can be found in `./utils_generation/construct_prompts.py`. This command will save hidden states to `generation_results` and will save zero-shot accuracy to `generation_results/generation_results.csv`.

3. To test our methods, run:

```bash
python extraction_main.py --model <MODEL> --prefix normal
```

For the example above, you can run:

```bash
python extraction_main.py --model roberta-large-mnli --prefix confusion  --datasets imdb amazon-polarity
```

Once finished, results will be saved in `extraction_results/{model}_{prefix}_{seed}.csv`, and the direction (`coef`) will be saved in `extraction_results/params`.


## Development

After installing the package with `pip install -e .[dev]`, please run `pre-commit install` to install the pre-commit hooks. This will automatically run `black`, `codespell`, and `flake8` on all the files you commit. You can also run `black .` and `flake8 .` to run these checks on all the files in the repository.

Please use type annotations for all new functions and methods. We're working on adding type annotations to the existing code, but it's a slow process.
