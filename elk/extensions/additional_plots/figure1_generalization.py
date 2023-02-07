# %%
from pathlib import Path
from elk.utils_evaluation.ccs import CCS
import numpy as np
import matplotlib.pyplot as plt
from elk.utils_evaluation.utils_evaluation import (
    get_hidden_states,
    get_permutation,
    normalize,
    split,
)
import torch


torch.autograd.set_grad_enabled(False)
# %%
parent_folder = Path(__file__).parent.parent.parent

hidden_states = get_hidden_states(
    hidden_states_directory=parent_folder / "generation_results",
    model_name="unifiedqa-t5-11b",
    dataset_name="copa",
    prefix="normal",
    language_model_type="encoder",
    layer=-1,
    mode="concat",
    num_data=1000,
)
permutation = get_permutation(hidden_states)
normalized_hidden_states = normalize(hidden_states, permutation, include_test_set=True)
# hidden_states = get_hidden_states(
#     hidden_states_directory=args.hidden_states_directory,
#     model_name=args.model,
#     dataset_name=args.dataset,
#     prefix=args.prefix,
#     language_model_type=args.language_model_type,
#     layer=args.layer,
#     mode=args.mode,
#     num_data=args.num_data,
# )
# permutation = get_permutation(hidden_states)
# normalized_hidden_states = normalize(
#     hidden_states, permutation, include_test_set=args.include_test_set
# )


features, labels = split(
    normalized_hidden_states,
    permutation,
    prompts=range(len(normalized_hidden_states)),
    split="test",
)

# %%
path = parent_folder / "trained" / "ccs_model.pt"
# ccs = CCS.load(path).to(args.device)
ccs = CCS.load(path).to("cuda")
# %%
# x0, x1 = torch.from_numpy(features).to(args.device).chunk(2, dim=1)
x0, x1 = torch.from_numpy(features).to("cuda").chunk(2, dim=1)


def predict(x):
    return torch.sigmoid(ccs(x)).cpu().numpy()[..., 0]


predictions0, predictions1 = predict(x0), predict(x1)
pred_probs = 0.5 * (predictions0 + (1 - predictions1))
true_predictions = pred_probs[labels == 1]
false_predictions = pred_probs[labels == 0]
# %%
plt.hist(true_predictions, bins=30, alpha=0.5, label="True", range=(0, 1))
plt.hist(false_predictions, bins=30, alpha=0.5, label="False", range=(0, 1))
plt.legend()
plt.title("Estimated probabilities for copa on unifiedqa-t5-11b")
# %%
