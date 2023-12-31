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
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lifelines.statistics import logrank_test

def km_curve(df,label,save_dir):
    '''
    Arguments:
        df {dataframe} -- Patient information including survival time, survival status, and classification labels
        cph_auc {np.array} -- AUC values at different time points for training set
        cph_auc_test {np.array} -- AUC values at different time points for test set
        label {str} -- Labels for each group and title name
        save_dir {str} -- Path for saving image
    '''
    mean_risk_factor,max_risk_factor,min_risk_factor  = df[label].median(),df[label].max(),df[label].min()
    df['risk_'] = pd.cut(df['risk_score'],[min_risk_factor, mean_risk_factor, max_risk_factor])
    fig, ax = plt.subplots()
    kmf = KaplanMeierFitter()
    itr = 0
    for name, grouped_df in df.groupby('risk_'):
        if itr == 0:
            target = 'Low Risk'
            group1 = grouped_df
        else:
            target = 'High Risk'
            group2 = grouped_df
        itr+= 1
        kmf.fit(grouped_df["survival_month"], grouped_df["status_dead"], label= target)
        kmf.plot_survival_function(ax=ax)
        plt.xlabel('survival months')
        plt.ylabel('Survival Probability')
        median = median_survival_times(kmf)
    ax.legend(loc='upper right')
    results = logrank_test(group1['survival_month'].values, group2['survival_month'].values, 
                       group1['status_dead'].values, group2['status_dead'].values)
    p_value = results.p_value
    plt.text(0.75, 0.8, f'p value: {p_value:.2e}', transform=plt.gca().transAxes, fontsize=10)
    plt.savefig(save_dir,dpi = 500, bbox_inches="tight")

def time_auc(va_times, cph_auc, cph_auc_test,label,save_dir):
    '''
    Arguments:
        va_times {np.array} -- Predicted Time Points
        cph_auc {np.array} -- AUC values at different time points for training set
        cph_auc_test {np.array} -- AUC values at different time points for test set
        label {str} -- Labels for each group and title name
        save_dir {str} -- Path for saving image
    '''
    train_label, test_label, title_name = label[0],label[1],label[2]
    cph_mean_auc = cph_auc.mean()
    cph_mean_auc_test = cph_auc_test.mean()
    plt.plot(va_times, cph_auc,color= "#82b0d2",marker="o",label= train_label + f"(mean AUC = {cph_mean_auc:.3f})")
    plt.plot(va_times, cph_auc_test,color= "#f9cc52",marker="o",label=test_label + f"(mean AUC = {cph_mean_auc_test:.3f})")
    plt.xlabel("months from enrollment", fontsize=14)
    plt.ylabel("time-dependent AUC", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlim(3, 40)  # 设置横坐标范围
    plt.ylim(0.5, 1)
    plt.title(title_name, fontsize=18)
    plt.savefig(save_dir,dpi = 500, bbox_inches="tight")