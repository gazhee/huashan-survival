import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from scipy import ndimage
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.img_processing import get_tumor
from torch.utils.data import TensorDataset, DataLoader
# resenet 18 34 50


class T1Dataset_regression(Dataset):
    def __init__(self,df_label,base_dir):
        self.dir = base_dir
        self.name = df_label['patient'].values.tolist()
        self.day = df_label['survival_month'].values.tolist()
        self.event = df_label['status_dead'].values.tolist()
    def __getitem__(self,index):
        patient = self.name[index]
        sub_dir = os.path.join(self.dir,patient+'_core.npy')
        day = self.day[index]
        event = self.event[index]
        img_data = np.load(sub_dir)
        img_data = torch.from_numpy(img_data)
        return img_data, day, event
    def __len__(self):
        return len(self.name)

class T1Dataset_regression_test(Dataset):
    def __init__(self,df_label,base_dir):
        self.dir = base_dir
        self.name = df_label['patient'].values.tolist()
        self.day = df_label['survival_month'].values.tolist()
        self.event = df_label['status_dead'].values.tolist()
    def __getitem__(self,index):
        patient = self.name[index]
        sub_dir = os.path.join(self.dir,patient+'_core.npy')
        day = self.day[index]
        event = self.event[index]
        img_data = np.load(sub_dir)
        img_data = torch.from_numpy(img_data)
        return img_data, day, event,patient
    def __len__(self):
        return len(self.name)


class T1Dataset_regression_mr_path(Dataset):
    def __init__(self,df_label,base_dir,base_dir_path):
        self.dir = base_dir
        self.pathdir = base_dir_path
        self.name = df_label['patient'].values.tolist()
        self.day = df_label['survival_month'].values.tolist()
        self.event = df_label['status_dead'].values.tolist()
    def __getitem__(self,index):
        patient = self.name[index]
        sub_dir = os.path.join(self.dir,patient+'_core.npy')
        path_dir = os.path.join(self.pathdir,patient,patient + '.npy')
        day = self.day[index]
        event = self.event[index]
        img_data = np.load(sub_dir)
        path_data = np.load(path_dir)
        img_data = torch.from_numpy(img_data)
        path_data = torch.from_numpy(path_data)
        return img_data,path_data, day, event
    def __len__(self):
        return len(self.name)

class T1Dataset_regression_mr_path_test(Dataset):
    def __init__(self,df_label,base_dir,base_dir_path):
        self.dir = base_dir
        self.pathdir = base_dir_path
        self.name = df_label['patient'].values.tolist()
        self.day = df_label['survival_month'].values.tolist()
        self.event = df_label['status_dead'].values.tolist()
    def __getitem__(self,index):
        patient = self.name[index]
        sub_dir = os.path.join(self.dir,patient+'_core.npy')
        path_dir = os.path.join(self.pathdir,patient,patient + '.npy')
        day = self.day[index]
        event = self.event[index]
        img_data = np.load(sub_dir)
        path_data = np.load(path_dir)
        img_data = torch.from_numpy(img_data)
        path_data = torch.from_numpy(path_data)
        return img_data,path_data, day, event,patient
    def __len__(self):
        return len(self.name)

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2 = 0.0):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).cuda()
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        return neg_log_loss