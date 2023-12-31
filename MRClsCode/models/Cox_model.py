import torch
import os
import h5py
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from models.SENet import Model_path_mr
from lifelines.utils import concordance_index
from models.MLP import MLP
import torch
import torchtuples as tt
from utils.img_processing import convert_path,normalize_min_max
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt

# Cox proportional hazards function
def calculate_param(df_test, model_cph,type = 0):
    '''
    Arguments:
        df_test {dataframe} -- Patient information table containing feature scores extracted by DLFE.
        model_cph -- Cox proportional hazards model
    Keyword Arguments:
        type {int} -- DLFE Model Type (default: {0})
            * type = 0 -- SE-DLFE-MRI
            * type = 1 -- SE-DLFE-H&E
            * type = 2 -- SE-DLFE
            * type = 3 -- SE-DLFE-M|H
    '''
    if type == 0:
        df_test[['SE_DLFE_MRI_score','WHO','Age','Gender','IDH']] = df_test[['SE_DLFE_MRI_score','WHO','Age','Gender','IDH']].astype('float32')
        feature = df_test[['SE_DLFE_MRI_score','WHO','Age','Gender','IDH']].values
    elif type == 1:
        df_test[['SE_DLFE_HE_score','WHO','Age','Gender','IDH']] = df_test[['SE_DLFE_HE_score','WHO','Age','Gender','IDH']].astype('float32')
        feature = df_test[['SE_DLFE_HE_score','WHO','Age','Gender','IDH']].values
    elif type == 1:
        df_test[['SE_DLFE_score','WHO','Age','Gender','IDH']] = df_test[['SE_DLFE_score','WHO','Age','Gender','IDH']].astype('float32')
        feature = df_test[['SE_DLFE_score','WHO','Age','Gender','IDH']].values
    elif type == 1:
        df_test[['SE_DLFE_HE_score','SE_DLFE_MRI_score','WHO','Age','Gender','IDH']] = df_test[['SE_DLFE_HE_score','SE_DLFE_MRI_score','WHO','Age','Gender','IDH']].astype('float32')
        feature = df_test[['SE_DLFE_HE_score','SE_DLFE_MRI_score','WHO','Age','Gender','IDH']].values
    get_target = lambda df_test: (df_test['survival_month'].values, df_test['status_dead'].values)
    durations_test_ex, events_test_ex = get_target(df_test)
    _ = model_cph.compute_baseline_hazards()
    surv = model_cph.predict_surv_df(feature)
    ev = EvalSurv(surv, durations_test_ex, events_test_ex, censor_surv='km')
    time_grid = np.linspace(durations_test_ex.min(), durations_test_ex.max(), 100)
    return ev.concordance_td(),ev.integrated_brier_score(time_grid)

if __name__ == '__main__':
    df_test = pd.DataFrame(os.path.join(''))
    in_features = 5
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False
    batch_size = 20
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    model_cph1 = CoxPH(net, tt.optim.Adam)
    c_index, brier_score = calculate_param(df_test,model_cph1)