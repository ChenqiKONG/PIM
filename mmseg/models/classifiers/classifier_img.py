import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Image_classifier(nn.Module):
    def __init__(self,):
        super(Image_classifier, self).__init__()
        self.sepconv1 = SeparableConv2d(96, 192, 1, 1, 0, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(384)
        self.conv3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = nn.Conv2d(in_channels=768*2, out_channels=768*2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(768*2)
        self.fc = nn.Linear(1536, 2)

    def forward(self, feature):
        f1, f2, f3, f4 = feature
        F1 = self.sepconv1(f1)
        F1 = self.bn1(F1)
        F1 = self.maxpool(F1)
        F2 = torch.cat((F1, f2), 1)
        F2 = self.conv2(F2)
        F2 = self.bn2(F2)
        F2 = self.maxpool(F2)
        F3 = torch.cat((F2, f3), 1)
        F3 = self.conv3(F3)
        F3 = self.bn3(F3)
        F3 = self.maxpool(F3)
        F4 = torch.cat((F3, f4), 1)
        F4 = self.conv4(F4)
        F4 = self.bn4(F4)
        x = F.adaptive_avg_pool2d(F4, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
if __name__ == "__main__":
    model = Image_classifier()
    device = torch.device('cpu')
    model = model.to(device)
    # print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    features = (torch.rand(size=(2, 96, 128, 128)), torch.rand(size=(2, 192, 64, 64)), torch.rand(size=(2, 384, 32, 32)), torch.rand(size=(2, 768, 16, 16)))
    #features = features.to(device)
    x = model(features)
    print(x.size())
    # segmap, boundary_map, recon_img = model(faces)
    # print(segmap.size())
    # print(boundary_map.size())
    # print(recon_img.size())
    # print(model)