import pandas as pd
import os
import numpy as np
import time

def get_directory(dataset_name_w_num, args):
	directory = f"{args.save_base_dir}/{args.model}_{dataset_name_w_num}_{args.prefix}_{args.token_place}"

	if args.tag != "":
		directory += "_{}".format(args.tag)

	return directory

	
def save_hidden_state_to_np_array(hidden_state, dataset_name_w_num, type_list, args):
	
	directory = get_directory(dataset_name_w_num, args)
	if not os.path.exists(directory):
		os.mkdir(directory)

	# hidden states is num_data * layers * dim
	# logits is num_data * vocab_size
	for (typ, array) in zip(type_list, hidden_state):
		if args.save_all_layers or "logits" in typ:
			np.save(os.path.join(directory, f"{typ}.npy"), array)
		else:
			# only save the last layers for encoder hidden states
			for idx in args.states_index:
				np.save(os.path.join(directory, "{typ}_{args.states_location}{idx}.npy"), array[:, idx,:])


def save_records_to_csv(records, args):
	file_path = os.path.join(args.save_base_dir, f"{args.save_csv_name}.csv")
	if not os.path.exists(file_path):
		all_results = pd.DataFrame(columns = ["time", "model", "dataset", "prompt_idx", "num_data", "population", "prefix", "cal_zeroshot", "cal_hiddenstates", "log_probs", "calibrated", "tag"])
	else:
		all_results = pd.read_csv(file_path)
		
	current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	for dic in records:
		dic["time"] = current_time        
		spliter = dic["dataset"].split("_")
		dic["dataset"], dic["prompt_idx"] = spliter[0], int(spliter[2][6:])
	
	all_results = all_results.append(records, ignore_index = True)
	all_results.to_csv(file_path, index = False)

	print(f"Successfully saved {len(records)} items in records to {file_path}")    


def print_elapsed_time(start, prefix, name_to_dataframe):
    total_samples = sum([len(dataframe) for dataframe in name_to_dataframe.values()])
    end = time.time()
    elapsed_minutes = round((end - start) / 60, 1)
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f"Prefix used: {prefix}, applied to {len(name_to_dataframe)} datasets, {total_samples} samples in total, and took {elapsed_minutes} minutes.")
    print("\n\n---------------------------------------\n\n")