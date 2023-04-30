import pickle
from pathlib import Path
import numpy as np

from torch import Tensor
import torch
from elk.metrics.eval import to_one_hot

from elk.metrics.roc_auc import roc_auc_ci

# imdb
root = Path('/home/laurito/elk-reporters/microsoft/deberta-large-mnli/imdb/quizzical-allen/transfer_eval')

# boolq
# root = Path('/home/laurito/elk-reporters/microsoft/deberta-large-mnli/imdb/quizzical-allen/transfer_eval')
# elk eval "microsoft/
# deberta-large-mnli/i
# mdb/quizzical-allen"
#  microsoft/deberta-l
# arge-mnli imdb --num
# _gpus 1             

# load pickle file
with open(root / 'vals.pkl', 'rb') as f:
    vals_buffers = pickle.load(f)

y_logits_means = []
y_trues_list = []
k_prompts_aurocs = []
for vals in vals_buffers:
    print("vals.shape", len(vals))

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

    k_prompts_aurocs.append(auroc)

    print("layer", vals[0]["layer"], "auroc", auroc)

def get_best_aurocs_indices(aurocs, max=5):
    sorted_indices = sorted(range(len(aurocs)), key=lambda i: aurocs[i].estimate)
    # the best aurocs are at the end of the list
    return sorted_indices[-max:]

best_aurocs_indices = get_best_aurocs_indices(k_prompts_aurocs)
print("best_aurocs_indices", best_aurocs_indices)

y_trues = [y_trues_list[i] for i in best_aurocs_indices]
y_logits = [y_logits_means[i] for i in best_aurocs_indices]

y_logits_layers = torch.stack(y_logits)
y_layer_logits_means = torch.mean(y_logits_layers, dim=0)

auroc = roc_auc_ci(y_trues[2], y_layer_logits_means[..., 1] - y_layer_logits_means[..., 0])
print(auroc)


