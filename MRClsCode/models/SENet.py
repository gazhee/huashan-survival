# 定义模型
import torch
import torch.nn as nn

class ResidualBlock1(nn.Module):
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
        out = identity + x
        return out

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



class Model(nn.Module):
    def __init__(self,num_classes=3):
        super(Model,self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels= 6,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding = 1)
        self.conv2_1 = ResidualBlock(in_channels=64,out_channels=64)
        self.conv2_2 = ResidualBlock(in_channels=64,out_channels=64)

        # conv3_x
        self.conv3_1 = ResidualBlock(in_channels=64,out_channels=128)
        self.conv3_2 = ResidualBlock(in_channels=128,out_channels=128)

        # conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128,out_channels=256)
        self.conv4_2 = ResidualBlock(in_channels=256,out_channels=256)

        # conv5_x
        self.conv5_1 = ResidualBlock(in_channels=256,out_channels=512)
        self.conv5_2 = ResidualBlock(in_channels=512,out_channels=512)

        # avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # fc
        self.fc = nn.Linear(in_features=512,out_features=num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

        # softmax 
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2_x
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        # conv3_x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        # conv4_x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # conv5_x
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        #avgpool+ fc + softmax
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        # x = self.bn(x)
        return x


class Model_path_mr(nn.Module):
    def __init__(self,input_dim=6, path_data_dim=1024,hidden_dim = 512, num_classes=1):
        super(Model_path_mr,self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels= input_dim,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding = 1)
        self.conv2_1 = ResidualBlock(in_channels=64,out_channels=64)
        self.conv2_2 = ResidualBlock(in_channels=64,out_channels=64)

        # conv3_x
        self.conv3_1 = ResidualBlock(in_channels=64,out_channels=128)
        self.conv3_2 = ResidualBlock(in_channels=128,out_channels=128)

        # conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128,out_channels=256)
        self.conv4_2 = ResidualBlock(in_channels=256,out_channels=256)

        # conv5_x
        self.conv5_1 = ResidualBlock(in_channels=256,out_channels=512)
        self.conv5_2 = ResidualBlock(in_channels=512,out_channels=512)

        # avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # fc
        self.fc1 = nn.Linear(hidden_dim + path_data_dim,100)
        self.fc2 = nn.Linear(100, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)


    def forward(self,x,path_data):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2_x
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        # conv3_x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        # conv4_x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # conv5_x
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        #avgpool+ fc + softmax
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = torch.cat((x, path_data), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x

    
if __name__ == '__main__':
    model =Model_path_mr()

    # 定义输入 [N,C,W,D]
    input = torch.ones([16,6,128,128])
    clin = torch.ones([16,1024])
    output = model(input,clin)
    print("output.shape = ",output.shape)