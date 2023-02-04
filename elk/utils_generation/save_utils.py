import pandas as pd
import os
import numpy as np
import time

def get_directory(save_base_dir, model_name, dataset_name_w_num, prefix, token_place, tags=None):
	"""
	Create a directory name given a model, dataset, number of data points and prefix.

	Args:
		save_base_dir (str): the base directory to save the hidden states
		model_name (str): the name of the model
		dataset_name_w_num (str): the name of the dataset with the number of data points
		prefix (str): the prefix
		token_place (str): Determine which token's hidden states will be generated. Can be `first` or `last` or `average`
		tags (list): an optional list of strings that describe the hidden state
	
	Returns:
		directory (str): the directory name	
	"""
	directory = f"{save_base_dir}/{model_name}_{dataset_name_w_num}_{prefix}_{token_place}"

	if tags != None:
		for tag in tags:
			directory += f"_{tag}"

	return directory

	
def save_hidden_state_to_np_array(hidden_state, dataset_name_w_num, type_list, args):
	"""
	Save the hidden states to a numpy array at the directory created by `get_directory`.

	Args:
		hidden_state (list): a list of hidden state arrays
		dataset_name_w_num (str): the name of the dataset with the number of data points
		type_list (list): a list of strings that describe the type of hidden state
		args (argparse.Namespace): the arguments

	Returns:
		None
	"""
	directory = get_directory(args.save_base_dir, args.model, dataset_name_w_num, args.prefix, args.token_place)
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
				np.save(os.path.join(directory, f"{typ}_{args.states_location}{idx}.npy"), array[:, idx,:])


def save_records_to_csv(records, args):
	"""
	Save the records to a csv file at the base directory + save csv name.

	Args:
		records (list): a list of dictionaries that contains metadata about the experiment
		args (argparse.Namespace): the arguments
	
	Returns:
		None
	"""
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
	"""
	Print information about the prefix used, the number of data points in a dataset, and the time elapsed.

	Args:
		start (float): the start time
		prefix (str): the prefix used
		name_to_dataframe (dict): a dictionary that maps dataset name to the dataframe
	
	Returns:
		None
	"""
	total_samples = sum([len(dataframe) for dataframe in name_to_dataframe.values()])
	end = time.time()
	elapsed_minutes = round((end - start) / 60, 1)
	print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
	print(f"Prefix used: {prefix}, applied to {len(name_to_dataframe)} dataset-prefix combinations, {total_samples} samples in total, and took {elapsed_minutes} minutes.")
	print("\n\n---------------------------------------\n\n")