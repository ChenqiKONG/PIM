import torch
import torch.nn as nn
import os
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_cls(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x