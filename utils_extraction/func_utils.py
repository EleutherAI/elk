import numpy as np


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])

def adder(df, model, prefix, method, prompt_level, train, test, accuracy, std, location, layer, loss):
    return df.append({
                "model": model, 
                "prefix": prefix, 
                "method": method, 
                "prompt_level": prompt_level, 
                "train": train, 
                "test": test, 
                "accuracy": accuracy,
                "std": std,
                "location": location, 
                "layer": layer, 
                "loss": loss,
            }, ignore_index=True)