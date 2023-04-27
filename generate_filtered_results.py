import pickle
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
import json
from tqdm import tqdm


# reinstall promptsourece

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side = "left")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# load the above model on to the specified GPU
device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
model.to(device)

with open("./counterfact.json", "rb") as f:
    counterfact_dataset = json.load(f)

stripped_dataset = []
for each in counterfact_dataset:
    target_true = each["requested_rewrite"]["target_true"]["str"]
    target_false = each["requested_rewrite"]["target_new"]["str"]
    labels = [target_true]#[target_false, target_true]
    for i, answer in enumerate(labels):
        row = {
            "subject": each["requested_rewrite"]["subject"],
            "proposition": each["requested_rewrite"]["prompt"].format(each["requested_rewrite"]["subject"]) + " " + answer,
            "subject+predicate": each["requested_rewrite"]["prompt"].format(each["requested_rewrite"]["subject"]),
            "answer": answer,
            "label": i,
            "case_id": each["case_id"],
        }
        stripped_dataset.append(row.copy())

tokenizer.pad_token = tokenizer.eos_token
def run_batch(texts):
    input_ids = tokenizer(texts, return_tensors='pt', padding=True).input_ids.to(device)
    # get attention mask
    attention_mask = input_ids.ne(tokenizer.eos_token_id).long().to(device)
    # model generate with attention masks
    output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask, max_length=30, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=5)
    # free up memory
    del input_ids
    return tokenizer.batch_decode(output, skip_special_tokens=True)

# split stripped_dataset into batches of 20
batched_dataset = [stripped_dataset[i:i + 20] for i in range(0, len(stripped_dataset), 20)]

results = []
for each in tqdm(batched_dataset):
    input_text = [x["subject+predicate"] for x in each]
    case_ids = [x["case_id"] for x in each]
    results.append({
        "case_id": case_ids,
        "completions": run_batch(input_text)})

pickle.dump(results, open("filtration_results.pkl", "wb"))
