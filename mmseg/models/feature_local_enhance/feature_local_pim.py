import torch 
import torch.nn as nn
import torch.nn.functional as F

## cd, ad, rd convolutions
def createConvFunc():

    def func_cdc(x, weights, bias=None, stride=1, padding=1, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
        assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
        assert padding == dilation, 'padding for cd_conv set wrong'

        weights_c = weights.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc

    def func_rdc(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
        assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
        padding = 2 * dilation

        shape = weights.shape
        if weights.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weights = weights.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y
    
    def func_all_dc(x, weights_cd, weights_rd):
        cdc_out = func_cdc(x, weights_cd, bias=None, stride=1, padding=1, dilation=1, groups=1)
        #adc_out = func_adc(x, weights_ad, bias=None, stride=1, padding=1, dilation=1, groups=1)
        rdc_out = func_rdc(x, weights_rd, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return torch.cat((cdc_out, rdc_out), 1)

    return func_all_dc


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


class feature_local_conv(nn.Module):
    def __init__(self, op_type=None):
        super(feature_local_conv, self).__init__()
        self.le_block = createConvFunc()
        self.weights_cd0 = nn.Parameter(torch.randn(96, 96, 3, 3))
        self.weights_cd1 = nn.Parameter(torch.randn(192, 192, 3, 3))
        self.weights_cd2 = nn.Parameter(torch.randn(384, 384, 3, 3))
        self.weights_cd3 = nn.Parameter(torch.randn(768, 768, 3, 3))

        self.weights_rd0 = nn.Parameter(torch.randn(96, 96, 3, 3))
        self.weights_rd1 = nn.Parameter(torch.randn(192, 192, 3, 3))
        self.weights_rd2 = nn.Parameter(torch.randn(384, 384, 3, 3))
        self.weights_rd3 = nn.Parameter(torch.randn(768, 768, 3, 3))

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

        

    def forward(self, feature):
        le_feature0 = self.le_block(feature[0], self.weights_cd0, self.weights_rd0)
        le_feature0 = self.weight_learn0(le_feature0)
        le_feature0 = self.norm0(le_feature0)
        le_feature0 = self.conv_sep_all0(le_feature0)
        le_feature_le0 = self.norm_all0(le_feature0) + feature[0]

        le_feature1 = self.le_block(feature[1], self.weights_cd1, self.weights_rd1)
        le_feature1 = self.weight_learn1(le_feature1)
        le_feature1 = self.norm1(le_feature1)
        le_feature1 = self.conv_sep_all1(le_feature1)
        le_feature_le1 = self.norm_all1(le_feature1) + feature[1]

        le_feature2 = self.le_block(feature[2], self.weights_cd2, self.weights_rd2)
        le_feature2 = self.weight_learn2(le_feature2)
        le_feature2 = self.norm2(le_feature2)
        le_feature2 = self.conv_sep_all2(le_feature2)
        le_feature_le2 = self.norm_all2(le_feature2) + feature[2]

        le_feature3 = self.le_block(feature[3], self.weights_cd3, self.weights_rd3)
        le_feature3 = self.weight_learn3(le_feature3)
        le_feature3 = self.norm3(le_feature3)
        le_feature3 = self.conv_sep_all3(le_feature3)
        le_feature_le3 = self.norm_all3(le_feature3) + feature[3]

        return [le_feature_le0, le_feature_le1, le_feature_le2, le_feature_le3]