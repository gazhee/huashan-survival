import torch
import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from models.SENet import Model_path_mr
from lifelines.utils import concordance_index
from models.MLP import MLP
import torch
import torchtuples as tt
from utils.img_processing import convert_path,normalize_min_max
import matplotlib.pyplot as plt
from sksurv.util import Surv
from sksurv.metrics import brier_score
from sklearn.metrics import brier_score_loss
from sksurv.svm import HingeLossSurvivalSVM


class SVM:
    def __init__(self,
                 alpha: float,
                 feature_col: str,
                 kernel_name: str,
                 max_iter: int):
        self.alpha = alpha
        self.kernel_name = kernel_name
        self.iter = max_iter
        self.feature_col = feature_col
    
    def dataset(self,df):
        feature = df[self.feature_col].values
        y_ = Surv.from_dataframe(
                            event='status_dead', 
                            time='survival_month', 
                            data= df)
        return feature,y_
    def make_model(self,feature,y_):
        """ This function is the model training.
        """
        model = HingeLossSurvivalSVM(alpha=self.alpha,
                                   max_iter=self.iter,
                                   kernel=self.kernel_name)
        model.fit(feature, y_)
        return model
    
    def calculate_param(self,model,df_test,type = 'cindex'):

        """ This function is the model evaluation.

            :param model: Survival Support Vector Machine
            :param df_test: Test patient information table.
            :param type:  Evaluation criteria or standards. default{cindex}
                        * cindex: C index
                        * risk: Rank samples according to survival times
                        * brier_loss: brier score
        """
        feature_test, y_test_ = self.dataset(df_test)
        if type == 'cindex':
            score = model.score(feature_test,y_test_)
            return score
        elif type == 'risk':
            score = model.preict(feature_test)
            return score
        elif type == 'brier_loss':
            risk_score = model.preict(feature_test)
            score = brier_score_loss(df_test['status_dead'], risk_score)
            return score
        