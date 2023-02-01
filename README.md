### Introduction

**WARNING**: This code is actively being hacked on and is probably unstable in all sorts of ways. Proceed with caution. If you'd like to help improve the code, please check out the #eliciting-latent-knowledge channel on the EleutherAI Discord server.

This code consists of two parts. First, you can use `generation_main.py` to calculate the zero-shot accuracy and generate hidden states. Second, you can use `extraction_main.py` to run our methods, `CCS`, `TPC` and `BSS`, to get final predictions and performance.


### Dependencies

Our code uses [PyTorch](http://pytorch.org) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index). You will also need to install [promptsouce](https://github.com/bigscience-workshop/promptsource), a toolkit for NLP prompts. We tested our code on Python 3.8.


### Quick **Start**

1. First install the package with `pip install -e .` in the root directory.
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
