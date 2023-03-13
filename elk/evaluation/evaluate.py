import csv
import os
import pickle
from dataclasses import dataclass
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Literal, Optional, cast

import torch
import torch.multiprocessing as mp
import yaml
from simple_parsing.helpers import Serializable, field
from torch import Tensor
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from datasets import DatasetDict
from elk.training.preprocessing import normalize

from ..extraction import ExtractionConfig, extract
from ..files import elk_reporter_dir, memorably_named_dir
from ..utils import assert_type, held_out_split, int16_to_float32, select_usable_devices


@dataclass
class EvaluateConfig(Serializable):
    target: ExtractionConfig
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    max_gpus: int = -1


def evaluate_reporter(
    cfg: EvaluateConfig,
    dataset: DatasetDict,
    layer: int,
    devices: list[str],
    world_size: int = 1,
):
    """Evaluate a single reporter on a single layer."""
    rank = os.getpid() % world_size
    device = devices[rank]

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    with dataset.formatted_as("torch", device=device, dtype=torch.int16):
        train, test = dataset["train"], held_out_split(dataset)
        test_labels = cast(Tensor, test["label"])

        _, test_h = normalize(
            int16_to_float32(assert_type(Tensor, train[f"hidden_{layer}"])),
            int16_to_float32(assert_type(Tensor, test[f"hidden_{layer}"])),
            method=cfg.normalization,
        )

    reporter_path = elk_reporter_dir() / cfg.source / "reporters" / f"layer_{layer}.pt"
    reporter = torch.load(reporter_path, map_location=device)
    reporter.eval()

    test_x0, test_x1 = test_h.unbind(dim=-2)

    test_result = reporter.score(
        (test_x0, test_x1),
        test_labels,
    )

    stats = [layer, *test_result]
    return stats


def evaluate_reporters(cfg: EvaluateConfig, out_dir: Optional[Path] = None):
    if cfg.source == 'zero-shot':
        print("Evaluating zero-shot performance.")
        
        evaluate_zeroshot(cfg, out_dir)

        return

    ds = extract(cfg.target, max_gpus=cfg.max_gpus)

    layers = [
        int(feat[len("hidden_") :])
        for feat in ds["train"].features
        if feat.startswith("hidden_")
    ]

    devices = select_usable_devices(cfg.max_gpus)
    num_devices = len(devices)

    transfer_eval = elk_reporter_dir() / cfg.source / "transfer_eval"
    transfer_eval.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_dir = memorably_named_dir(transfer_eval)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print the output directory in bold with escape codes
    print(f"Saving results to \033[1m{out_dir}\033[0m")

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    cols = ["layer", "loss", "acc", "cal_acc", "auroc"]
    # Evaluate reporters for each layer in parallel
    with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
        fn = partial(
            evaluate_reporter, cfg, ds, devices=devices, world_size=num_devices
        )
        writer = csv.writer(f)
        writer.writerow(cols)

        mapper = pool.imap if num_devices > 1 else map
        for i, *stats in tqdm(mapper(fn, layers), total=len(layers)):
            writer.writerow([i] + [f"{s:.4f}" for s in stats])

    print("Results saved")


def evaluate_zeroshot(cfg: EvaluateConfig, out_dir: Optional[Path] = None):
    ds = extract(cfg.target, max_gpus=cfg.max_gpus)

    zs_eval = elk_reporter_dir() / cfg.source / "zero_shot_eval"
    zs_eval.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_dir = memorably_named_dir(zs_eval)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print the output directory in bold with escape codes
    print(f"Saving results to \033[1m{out_dir}\033[0m")

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    def get_logprobs_tensor(split):
        print(f"Loading {split} predictions from disk")

        # returns (num_examples, num_prompts, num_labels)
        logprobs_lst = []
        for ex in tqdm(ds[split]):
            logprobs_lst.append(ex['losses'])
        logprobs = -torch.tensor(logprobs_lst)
        return logprobs

    train_logprobs = get_logprobs_tensor('train')
    test_logprobs = get_logprobs_tensor('test')

    print(test_logprobs.shape)

    num_prompts = test_logprobs.shape[1]

    print(num_prompts)

    # calculate correction factor gamma
    # l_pos - l_neg: (num_examples, num_prompts)
    diff = train_logprobs[:, :, 1] - train_logprobs[:, :, 0]

    # there are an equal number of elements below and above gamma: (num_prompts,)
    gammas, _ = torch.median(diff.T, dim=-1, keepdim=False)

    # add correction factor
    test_logprobs_cal = test_logprobs.clone()
    test_logprobs_cal[:, :, 0] += gammas.unsqueeze(0)

    # get predictions (num_examples, num_prompts)
    zs_preds = torch.argmax(test_logprobs, dim=-1)
    zs_preds_cal = torch.argmax(test_logprobs_cal, dim=-1)

    # labels: (num_examples,) -> (num_examples, num_prompts)
    labels = torch.tensor(ds['test']['label']).unsqueeze(1)
    labels = labels.repeat(1, num_prompts)

    # accuracy calculation
    raw_acc = zs_preds.flatten().eq(labels.flatten()).float().mean()
    cal_acc = zs_preds_cal.flatten().eq(labels.flatten()).float().mean()

    # get aggregate truth probability
    # (1 - e^l_neg) * 0.5 + (e^l_pos) * 0.5
    truth_prob = ((1 - torch.exp(test_logprobs_cal[:, :, 0])) + torch.exp(train_logprobs[:, :, 1])) * 0.5

    # auroc calculation
    # logprobs -> probs
    test_probs = torch.exp(test_logprobs)
    auroc = roc_auc_score(labels.flatten(), truth_prob.flatten())

    print('Raw accuracy: %.4f\nCalibrated accuracy:%.4f\nAUROC: %.4f' % (raw_acc, cal_acc, auroc))    

    cols = ['acc', 'cal_acc', 'auroc']

    with open(out_dir / "eval.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerow([raw_acc, cal_acc, auroc])
