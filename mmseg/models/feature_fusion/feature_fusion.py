import torch 
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class weight_block(nn.Module):
    def __init__(self, channel):
        super(weight_block, self).__init__()
        self.fc = nn.Linear(channel, 2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, feature):
        x = F.adaptive_avg_pool2d(feature, (1, 1))
        x = self.fc(torch.squeeze(x))
        x = self.relu(x)
        weight = nn.functional.sigmoid(x)
        weight0 = torch.unsqueeze(weight[:,0], 1)
        weight1= torch.unsqueeze(weight[:,1], 1)
        weights = torch.cat((torch.tile(weight0, (1,int(feature.size(1)/2))), torch.tile(weight1, (1,int(feature.size(1)/2)))), dim=1)
        weights = torch.unsqueeze(weights, 2)
        weights = torch.unsqueeze(weights, 3)
        feature = feature*weights.view(weights.size(0), weights.size(1), 1, 1)
        return feature


class feature_fusion(nn.Module):
    def __init__(self, op_type=None):
        super(feature_fusion, self).__init__()

        self.weight_learn0 = weight_block(96*2)
        self.weight_learn1 = weight_block(192*2)
        self.weight_learn2 = weight_block(384*2)
        self.weight_learn3 = weight_block(768*2)

        self.norm0 = nn.BatchNorm2d(96*2)
        self.norm1 = nn.BatchNorm2d(192*2)
        self.norm2 = nn.BatchNorm2d(384*2)
        self.norm3 = nn.BatchNorm2d(768*2)
        
        self.conv_sep_all0 = SeparableConv2d(96*2, 96, 3, 1, 1)
        self.norm_all0 = nn.BatchNorm2d(96)

        self.conv_sep_all1 = SeparableConv2d(192*2, 192, 3, 1, 1)
        self.norm_all1 = nn.BatchNorm2d(192)

        self.conv_sep_all2 = SeparableConv2d(384*2, 384, 3, 1, 1)
        self.norm_all2 = nn.BatchNorm2d(384)

        self.conv_sep_all3 = SeparableConv2d(768*2, 768, 3, 1, 1)
        self.norm_all3 = nn.BatchNorm2d(768)

        

    def forward(self, feature_dc, feature_rc):
        feature = [torch.cat((feature_dc[0], feature_rc[0]), 1), torch.cat((feature_dc[1], feature_rc[1]), 1), 
                    torch.cat((feature_dc[2], feature_rc[2]), 1), torch.cat((feature_dc[3], feature_rc[3]), 1)]

        feature0 = self.weight_learn0(feature[0])
        feature0 = self.norm0(feature0)
        feature0 = self.conv_sep_all0(feature0)
        feature_fus0 = self.norm_all0(feature0) + feature_dc[0]

        feature1 = self.weight_learn1(feature[1])
        feature1 = self.norm1(feature1)
        feature1 = self.conv_sep_all1(feature1)
        feature_fus1 = self.norm_all1(feature1) + feature_dc[1]

        feature2 = self.weight_learn2(feature[2])
        feature2 = self.norm2(feature2)
        feature2 = self.conv_sep_all2(feature2)
        feature_fus2 = self.norm_all2(feature2) + feature_dc[2]

        feature3 = self.weight_learn3(feature[3])
        feature3 = self.norm3(feature3)
        feature3 = self.conv_sep_all3(feature3)
        feature_fus3 = self.norm_all3(feature3) + feature_dc[3]

        return [feature_fus0, feature_fus1, feature_fus2, feature_fus3]