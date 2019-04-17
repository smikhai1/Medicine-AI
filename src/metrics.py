import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from  albumentations import *
import torch
import pandas as pd
from scipy.ndimage import label


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

class map3(nn.Module):
    def __init__(self, ):
        super(map3, self).__init__()
        
    def forward(self, preds, targs):
        # targs = np.where(targs==1)[1]
        predicted_idxs = preds.sort(descending=True)[1]
        top_3 = predicted_idxs[:, :3]
        res = mapk([[t] for t in targs.cpu().numpy()], top_3.cpu().numpy(), 3)
        return -torch.tensor(res)