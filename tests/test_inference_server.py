import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from elk.extraction.inference_server import InferenceServer


@pytest.mark.filterwarnings("ignore:Unable to find a decoding function")
def test_inference_server():
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_dict({"text": ["Lorem", "ipsum", "dolor", "sit", "amet"]})

    ds = ds.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
        batched=True,
        remove_columns=["text"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    with ds.formatted_as("torch"):

        def gt_inference(ex: dict):
            with torch.no_grad():
                ex = {k: v.to(device).unsqueeze(0) for k, v in ex.items()}
                return {"out": model(**ex)}

        gt_out_ds = ds.map(gt_inference, batched=False, remove_columns=ds.column_names)
        gt_outs = gt_out_ds["out"]

    def test_config(fsdp: bool, num_workers: int, cpu_offload: bool = True):
        with InferenceServer(
            model_str=model_name,
            fsdp=fsdp,
            num_workers=num_workers,
            cpu_offload=cpu_offload,
        ) as server:
            outs = server.map_forward(ds)
            assert len(outs) == len(gt_outs)
            for out, gt_out in zip(outs, gt_outs):
                out_logits = out["logits"]
                assert torch.allclose(out_logits, gt_out["logits"].cpu())

    test_config(fsdp=False, num_workers=-1)
    test_config(fsdp=False, num_workers=1)
    test_config(fsdp=False, num_workers=2)
    test_config(fsdp=True, num_workers=-1, cpu_offload=False)
    test_config(fsdp=True, num_workers=-1)
    test_config(fsdp=True, num_workers=1)
    test_config(fsdp=True, num_workers=2)
