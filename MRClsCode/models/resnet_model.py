import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score

class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        out = self.relu(x + residual)
        return out
    
class Resnet(nn.Module):
    def __init__(self, in_channels = 2, outputs = 3):
        super(Resnet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels= 16,kernel_size=7,stride=1,padding=3,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResidualBlock(in_channels=16)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=7,stride=1,padding=3,bias=False),
            ResidualBlock(in_channels=32)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7,stride=1,padding=3,bias=False),
            ResidualBlock(in_channels=64),
        )
        self.block5 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=64,out_features=outputs)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        if self.training:
            return x
        else:
            return self.softmax(x)

    
if __name__ == "__main__": 
    model = Resnet()
    model.train()
    input1 = torch.randn(16,3,128,128)
    # y = torch.tensor([0,1])
    # criterion = nn.CrossEntropyLoss()
    output = model(input1)
    # loss = criterion(output,y)
    print("output size:",output.size()) 