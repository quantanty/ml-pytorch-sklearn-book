import torch
from torch import nn
import torch.nn.functional as F

class ImageCNNv1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.gpool = nn.AvgPool2d((8, 8))
        self.fc1 = nn.Linear(256, 64)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        assert x.size(2) == 64 and x.size(3) == 64

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.gpool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.fc2(x))
        
        return x
    
class ImageCNNv2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.gpool = nn.AvgPool2d((4, 4))
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 16)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        assert x.size(2) == 64 and x.size(3) == 64

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.gpool(x)
        x = self.drop1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.sigmoid(self.fc3(x))
        
        return x
    
class ImageCNNv3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.gpool = nn.AvgPool2d((4, 4))
        self.fc1 = nn.Linear(512, 128)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        assert x.size(2) == 64 and x.size(3) == 64

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.gpool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.sigmoid(self.fc3(x))
        
        return x
    
class ImageCNNv4(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.gpool = nn.AvgPool2d((4, 4))
        self.fc1 = nn.Linear(512, 256)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        assert x.size(2) == 64 and x.size(3) == 64

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gpool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        x = F.sigmoid(self.fc4(x))
        
        return x