import numpy as np


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])

def append_data_to_df(df, model, prefix, method, prompt_level, train, test, accuracy, std, language_model_type, layer, loss):
    return df.append({
                "model": model, 
                "prefix": prefix, 
                "method": method, 
                "prompt_level": prompt_level, 
                "train": train, 
                "test": test, 
                "accuracy": accuracy,
                "std": std,
                "language_model_type": language_model_type, 
                "layer": layer, 
                "loss": loss,
            }, ignore_index=True)


def populate_evaluation_results(dataset_to_accurary_per_prompt, 
                    dataset_to_loss_per_prompt, 
                    stats_df, 
                    model, 
                    prefix, 
                    method, 
                    args, 
                    train_set, 
                    dataset_names):

    for dataset in dataset_names:
        if args.prompt_save_level == "all":
            stats_df = append_data_to_df(stats_df, model, prefix, method, "all", train_set, dataset,
                        accuracy = np.mean(dataset_to_accurary_per_prompt[dataset]),
                        std = np.std(dataset_to_accurary_per_prompt[dataset]),
                        language_model_type = args.language_model_type,
                        layer = args.layer,
                        loss = np.mean(dataset_to_loss_per_prompt[dataset]) if method in ["Prob", "BSS"] else ""
                        )
        else:
            for idx in range(len(dataset_to_accurary_per_prompt[dataset])):
                stats_df = append_data_to_df(stats_df, model, prefix, method, idx, train_set, dataset,
                            accuracy = dataset_to_accurary_per_prompt[dataset][idx],
                            std = "",
                            language_model_type = args.language_model_type,
                            layer = args.layer,
                            loss = dataset_to_loss_per_prompt[dataset][idx] if method in ["Prob", "BSS"] else ""
                            )

    return stats_df
