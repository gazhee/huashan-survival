import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
    
    
class Dataset_pathology_test(Dataset):
    def __init__(self,df_label,base_dir_path):
        self.pathdir = base_dir_path
        self.name = df_label['patient'].values.tolist()
        self.day = df_label['survival_month'].values.tolist()
        self.event = df_label['status_dead'].values.tolist()
    def __getitem__(self,index):
        patient = self.name[index]
        path_dir = os.path.join(self.pathdir,patient,patient + '.npy')
        day = self.day[index]
        event = self.event[index]
        path_data = np.load(path_dir)
        path_data = torch.from_numpy(path_data)
        return path_data, day, event,patient
    def __len__(self):
        return len(self.name)
    

class Dataset_mr_test(Dataset):
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


class Dataset_mr_pathology_test(Dataset):
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
