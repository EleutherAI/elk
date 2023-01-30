import numpy as np


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])

