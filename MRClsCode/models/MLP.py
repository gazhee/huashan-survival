import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channel = 1024, out_channel = 1):
        super(MLP, self).__init__()
        self.drop = 0.3
        self.norm = False
        self.dims = [in_channel,500, 80, out_channel]
        self.activation = 'ReLU'
        self.model = self._build_network()

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None:
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1],bias=False))
            if self.norm:
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            if i < 2:
                layers.append(eval('nn.{}()'.format(self.activation))) 
        print(layers)
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

