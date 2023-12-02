import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self,training = True):
        super(Net1, self).__init__()
        self.training = training
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 6,out_channels= 8, kernel_size = 7,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1,bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1,bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 7,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1,bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 7,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,7,bias= False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,7,bias= False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,256,7,bias= False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256,256,7,bias= False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(0.3)
        '''avgpool'''
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(256 * 1 * 1, 100),
            nn.Linear(100, 1)
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x= self.drop(x)
        x = self.conv2(x)
        x= self.drop(x)
        x = self.conv3(x)
        x= self.drop(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute channel-wise attention weights
        avg_out = self.avg_pool(x).squeeze(-1).squeeze(-1)
        channel_attention = self.fc(avg_out).unsqueeze(-1).unsqueeze(-1)

        # Apply attention to the input feature map
        return x * channel_attention

class Net2(nn.Module):
    def __init__(self,num_class  = 3):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 2,out_channels= 8, kernel_size = 3,bias=False),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3,bias=False),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3,bias=False),
            ChannelAttention(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,bias= False),
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 0),
            ChannelAttention(1024),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.drop = nn.Dropout(0.3)
        '''avgpool'''
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 100),
            nn.Linear(100, num_class)
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x= self.drop(x)
        x = self.conv2(x)
        x= self.drop(x)
        x = self.conv3(x)
        x= self.drop(x)
        x = self.conv4(x)
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Net3(nn.Module):
    def __init__(self,training = True, num_class = 3):
        super(Net3, self).__init__()
        self.training = training
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 2,out_channels= 8, kernel_size = 3,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 7,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 7,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3,bias=False),
            ChannelAttention(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3,bias=False),
            ChannelAttention(128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3,bias=False),
            ChannelAttention(256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 7,2, bias=False),
            # ChannelAttention(512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 7,bias=False),
            # ChannelAttention(512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 1024, 3,bias=False),
            # ChannelAttention(512),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
        )
        self.drop = nn.Dropout(0.3)
        '''avgpool'''
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 100),
            nn.Linear(100, num_class)
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x= self.drop(x)
        x = self.conv2(x)
        x= self.drop(x)
        x = self.conv3(x)
        x= self.drop(x)
        x = self.conv4(x)
        x= self.drop(x)
        x = self.drop(x)
        x= self.drop(x)
        x = self.conv5(x)
        x= self.drop(x)
        x = self.conv6(x)
        x= self.drop(x)
        x = self.conv7(x)
        x= self.drop(x)
        x = self.conv8(x)
        x= self.drop(x)
        x = self.conv9(x)
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Net4(nn.Module):
    def __init__(self,training = True):
        super(Net4, self).__init__()
        self.training = training
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 2,out_channels= 8, kernel_size = 3,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
        )

        self.drop = nn.Dropout(0.3)
        '''avgpool'''
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 100),
            nn.Linear(100, 5)
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x= self.drop(x)
        x = self.conv2(x)
        
        print(x.shape)
        # x = self.adaptive_avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


class SELayer(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=channel, out_features=channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=channel // reduction, out_features=channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        identity = x
        # [N,C,H,W]
        n,c,_,_ = x.size()
        x = self.avg_pool(x)
        x = x.view(n,c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(n,c,1,1)
        x = identity * x
        return x

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResidualBlock,self).__init__()
       
        self.channels_equal_flag = True
        if in_channels == out_channels:
            self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        else:
            self.channels_equal_flag = False
            self.conv1x1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1, stride = 2)
            self.bn1x1 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride = 2)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.selayer = SELayer(channel=out_channels)

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.channels_equal_flag == True:
            pass
        else:
            identity = self.conv1x1(identity)
            identity = self.bn1x1(identity)
            identity = self.relu(identity)
        x = self.selayer(x)
        out = identity + x
        return out



class SECNN(nn.Module):
    def __init__(self,in_channels = 2, num_classes = 5):
        super(SECNN, self).__init__()
        self.extract_features = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size= 7, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResidualBlock(in_channels=64,out_channels=128)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.out = nn.Sequential(
            nn.Linear(128 * 1 * 1, 64),
            nn.Linear(64,32),
            nn.Linear(32, num_classes)
        )
    def forward(self,x):
        x = self.extract_features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = self.out(x)
        return 





# heatmap regression 回归 1d 高斯
#  CAM
#  MSE 



if __name__ == "__main__":
    model = Net3()
    input1 = torch.randn(2,2,128,256)
    y = torch.tensor([0,1])
    criterion = nn.CrossEntropyLoss()
    output = model(input1)
    # loss = criterion(output,y)
    print(output.shape)