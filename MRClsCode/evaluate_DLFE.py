import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.Dataset import Dataset_mr_pathology_test,Dataset_mr_test,Dataset_pathology_test
from torch.utils.data import Dataset, DataLoader
from models.SENet import Model_path_mr,Model_mr
from models.MLP import MLP
import torch

def Test_SE_DLFE(test_dl,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        for itr,(images,images_path,day,event,name) in enumerate(test_dl):
            name = np.array(name).tolist()
            images, images_path = images.to(device), images_path.to(device)
            risk_score = model(images.float(),images_path.float()).squeeze()
            if itr == 0:
                deep_score_test = risk_score.cpu()
                patient_test = name
            else:
                deep_score_test = torch.cat((deep_score_test, risk_score.cpu()),0)
                patient_test = patient_test + name
    state_test = {
    'patient': patient_test,
    'score': deep_score_test
    }
    df_test_ = pd.DataFrame(state_test)
    return df_test_

def Test_SE_DLFE_H(test_dl,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        for itr,(images,day,event,name) in enumerate(test_dl):
            name = np.array(name).tolist()
            images = images.to(device)
            risk_score = model(images.float()).squeeze()
            if itr == 0:
                deep_score_test = risk_score.cpu()
                patient_test = name
            else:
                deep_score_test = torch.cat((deep_score_test, risk_score.cpu()),0)
                patient_test = patient_test + name
    state_test = {
    'patient': patient_test,
    'score': deep_score_test
    }
    df_test_ = pd.DataFrame(state_test)
    return df_test_

def Test_SE_DLFE_M(test_dl,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        for itr,(images,day,event,name) in enumerate(test_dl):
            name = np.array(name).tolist()
            images = images.to(device)
            risk_score = model(images.float()).squeeze()
            if itr == 0:
                deep_score_test = risk_score.cpu()
                patient_test = name
            else:
                deep_score_test = torch.cat((deep_score_test, risk_score.cpu()),0)
                patient_test = patient_test + name
    state_test = {
        'patient': patient_test,
        'score': deep_score_test
    }
    df_test_ = pd.DataFrame(state_test)
    return df_test_

if __name__ == '__main__':
    image_dir = ['MRClsCode/data/mr/','MRClsCode/data/pathology/']
    df_test = pd.read_csv('MRClsCode/data/info.csv')
    # i = 0 SE_DLFE; i = 1 SE_DLFE_M; i = 2 SE_DLFE_H
    i = 0
    if i == 0:
        model = Model_path_mr()
        test_ds = Dataset_mr_pathology_test(df_test,image_dir[0],image_dir[1])
        test_dl = DataLoader(
                        test_ds,
                        batch_size = 1,
        )
        df = Test_SE_DLFE(test_dl,model)
    elif i == 1:
        model = Model_mr(num_classes=1)
        test_ds = Dataset_mr_test(df_test,image_dir[0])
        test_dl = DataLoader(
                        test_ds,
                        batch_size = 1,
        )
        df = Test_SE_DLFE_M(test_dl,model)
    elif i == 2:
        model = MLP()
        test_ds = Dataset_pathology_test(df_test,image_dir[1])
        test_dl = DataLoader(
                        test_ds,
                        batch_size = 1,
        )
        df = Test_SE_DLFE_H(test_dl,model)
    
