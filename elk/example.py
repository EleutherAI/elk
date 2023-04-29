import pickle
from pathlib import Path
import numpy as np

from torch import Tensor
import torch
from elk.metrics.eval import to_one_hot

from elk.metrics.roc_auc import roc_auc_ci

root = Path('/home/laurito/elk-reporters/microsoft/deberta-large-mnli/imdb/quizzical-allen/transfer_eval')

# load pickle file
with open(root / 'vals.pkl', 'rb') as f:
    vals_buffers = pickle.load(f)

y_logits_means = []
y_trues_list = []
for vals in vals_buffers:
    y_logits = vals[0]["val_credences"]
    y_trues = vals[0]["val_gt"]
    (n, v, c) = y_logits.shape
    assert y_trues.shape == (n,)

    y_logits = y_logits.mean(dim=1)
    
    y_logits_means.append(y_logits)
    y_trues_list.append(y_trues)

    if c == 2:
        auroc = roc_auc_ci(y_trues, y_logits[..., 1] - y_logits[..., 0])
    else:
        auroc = roc_auc_ci(to_one_hot(y_trues, c).long(), y_logits)

    print("layer", vals[0]["layer"], "auroc", auroc)


y_trues = y_trues_list[22:-1]
y_logits = y_logits_means[22:-1]

layer_mean = torch.mean(torch.stack(y_logits), dim=2)

breakpoint()

i = 0
for y_logits, y_true in zip(y_logits_means, y_trues):
    auroc = roc_auc_ci(y_true, layer_mean[..., 1] - layer_mean[..., 0])
    print("auroc", auroc)
    i = i + 1
