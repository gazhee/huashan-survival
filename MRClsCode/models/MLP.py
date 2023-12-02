import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from sklearn.preprocessing import scale
from sksurv.datasets import load_flchain
from sksurv.util import Surv
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class MLP(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, in_channel = 1024, out_channel = 1):
        super(MLP, self).__init__()
        # parses parameters of network from configuration
        self.drop = 0.3
        self.norm = False
        self.dims = [in_channel,500, 80, out_channel]
        self.activation = 'ReLU'
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1],bias=False))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            if i < 2:
                layers.append(eval('nn.{}()'.format(self.activation))) 
        # layers.append(nn.Softmax(dim=0))
        print(layers)
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


if __name__ == "__main__":
    model = MLP()
    print(model)
    input1 = torch.randn(16,1024)
    output = model(input1)
    print(output)

