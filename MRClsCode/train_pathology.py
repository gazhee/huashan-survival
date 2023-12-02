
import torch
import os
import h5py
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader,Dataset
from lifelines.utils import concordance_index
import torch.nn as nn
from utils.PathDataset import Pathology_Dataset
from models.MLP import MLP
from pycox.models.loss import CoxPHLoss

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def fit(num_epochs,model,trainloader,testloader,i):
    base_dir = 'G:/glioma/final_data/glioma_survive/' 
    min_loss   = 4
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for x,y,e in trainloader:
            y = torch.as_tensor(y,dtype=torch.float64)
            e = torch.as_tensor(e, dtype=torch.long)
            x, y, e = x.to(device),y.to(device), e.to(device)
            x = x.type(torch.cuda.FloatTensor)
            risk_pred = model(x).squeeze()
            train_loss = criterion(risk_pred,y, e)
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            with torch.no_grad():
                running_loss += train_loss.item()
            epoch_loss = running_loss / len(trainloader)
        print('{} Loss: {:.8f}'.format('train',epoch_loss)) 
        running_loss_test = 0
        model.eval()
        for x,y,e in testloader:
            y = torch.as_tensor(y,dtype=torch.float64)
            e = torch.as_tensor(e, dtype=torch.long)
            x, y, e = x.to(device),y.to(device), e.to(device)
            x = x.type(torch.cuda.FloatTensor)
            risk_pred = model(x).squeeze()
            test_loss = criterion(risk_pred,y, e)
            with torch.no_grad():
                running_loss_test += test_loss.item()
            epoch_loss_test = running_loss_test / len(testloader)
        print('{} Loss: {:.8f}'.format('test',epoch_loss_test)) 
        if  epoch_loss_test < min_loss:
            min_loss = epoch_loss_test
            print('{} Min Loss: {:.8f}'.format('test',min_loss))
            state = {
                  'state_dict': model.state_dict(),
                  'optimizer' : optim.state_dict()
                }
            if i == 0:
                torch.save(state, os.path.join(base_dir ,'only_path',"survive_cnn_min_loss_only_path_TCGA1.pth"))
            elif i == 1:
                torch.save(state, os.path.join(base_dir ,'only_path',"survive_cnn_min_loss_only_path_huashan1.pth"))
    return model,val_acc_history, train_acc_history, valid_losses, train_losses

if __name__ == "__main__":
    base_dir = ['G:/glioma/final_data/huashan_MR/npy_file/TCGA_npy','G:/glioma/final_data/huashan_MR/npy_file/glioma_huashan_npy']
    base_dir_path = ['G:/glioma/final_data/huashan_MR/pathology/TCGA','G:/glioma/final_data/huashan_MR/pathology/huashan']
    df_dir = ['G:/glioma/final_data/glioma_survive/TCGA.csv','G:/glioma/final_data/glioma_survive/huashan.csv']
    i = 1
    df = pd.read_csv(df_dir[i])
    df_train, df_test, y_train, y_test = train_test_split(df,df['patient'],test_size=0.2,random_state=2022)
    train_ds = Pathology_Dataset(df_train,base_dir_path[i])
    test_ds = Pathology_Dataset(df_test,base_dir_path[i])
    train_dl = DataLoader(
                        train_ds,
                        shuffle= False,
                        batch_size = 20,
    )
    test_dl = DataLoader(
                        test_ds,
                        shuffle= False,
                        batch_size = 20,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP()
    criterion = CoxPHLoss().to(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(),lr=0.0001) 
    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses = fit(
                                                                                120, 
                                                                                model,
                                                                                train_dl,
                                                                                test_dl,
                                                                                i)