from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


###### SENet:Squeeze-and-Excitation Networks ######
class SE(nn.Module):
    def __init__(self, in_chs, reduction=16, mode='fc'):
        super().__init__()
        self.mode = mode
        self.chs = in_chs // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_chs, self.chs, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.chs, in_chs, bias=False),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
                nn.Conv2d(in_chs, self.chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.chs), nn.ReLU(inplace=True),
                nn.Conv2d(self.chs, in_chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_chs), nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.avg_pool(x)
        if self.mode == 'conv':
            out = self.conv(out)
        elif self.mode == 'fc':
            B, C, H, W = x.size()
            out = self.fc(out).reshape(B, C, 1, 1)
        else:
            raise ValueError("mode must be 'conv' or 'fc'.")
        return x * out  # Channel Attention


###### CBAM:Convolutional Block Attention Module ######
class CBAM(nn.Module):
    def __init__(self, in_chs, rotio, kernel_size):
        super().__init__()
        self.ca = ChannelAttention(in_chs, rotio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x  # Channel + Spatial Attention

class ChannelAttention(nn.Module):  # Channel Attention
    def __init__(self, in_chs, rotio=16):
        super().__init__()
        self.chs = in_chs // rotio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, self.chs, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.chs), nn.ReLU(),
            nn.Conv2d(self.chs, in_chs, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_chs),
        )

    def forward(self, x):
        avgout = self.conv(self.avg_pool(x))
        maxout = self.conv(self.max_pool(x))
        return F.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):  # Spatial Attention
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False),

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        return F.sigmoid(self.conv(x))


###### SKNet:Selective Kernel Networks ######
class SK(nn.Module):
    def __init__(self, in_chs, M, r, group, stride=1, L=32):
        super().__init__()
        d = np.max((in_chs // r, L))
        self.M = M
        self.in_chs = in_chs

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chs, in_chs, kernel_size=3+2*i, stride=stride, padding=1+i, groups=group),
                # nn.Conv2d(in_chs, in_chs, kernel_size=3+2*i, stride=stride+i, padding=1, groups=group),
                nn.BatchNorm2d(in_chs), nn.ReLU(inplace=True)
            ) for i in range(M)
        ])
        self.fc = nn.Linear(in_chs, d)
        self.fcs = nn.ModuleList([nn.Linear(d, in_chs) for i in range(M)])

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            feas = torch.cat([feas, fea], dim=1) if i > 0 else fea 

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            attenion = torch.cat([attenion, vector], dim=1) if i > 0 else vector

        attenion = F.softmax(attenion, dim=1).unsqueeze(-1).unsqueeze(-1)
        fea_V = (feas * attenion).sum(dim=1)
        return fea_V


###### ECANet:Efficient Channel Attention for Deep Convolutional Neural Networks ######
class ECA(nn.Module):
    def __init__(self, in_chs, gamma=2, b=1, ksize=None):
        super().__init__()
        self.ksize = np.abs((np.log2(in_chs) + b) // gamma)
        self.ksize = self.ksize if ksize is None else ksize
        self.ksize = self.ksize if self.ksize % 2 else self.ksize + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.ksize, padding=(self.ksize - 1) // 2, bias=False)
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        # y = y.expand_as(x)
        return x * y

if __name__ == '__main__':
    
    inputData = torch.randn([2, 64, 32, 32])
    print(f"Input Data Shape is {inputData.shape}")
    
    # model = SE(in_chs=64, reduction=16, mode='conv')
    # model = SK(in_chs=64, M=3, r=2, group=16)
    model = ECA(in_chs=64)
    y = model(inputData)
    print(y.shape)
