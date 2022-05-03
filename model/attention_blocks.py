__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction=16):
        nn.Module.__init__(self)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                nn.Sigmoid())

    def forward(self, x):
        x_init = x
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_init * y.expand_as(x_init)


class RecursiveSqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction=16, rec=3):
        nn.Module.__init__(self)
        self.rec = rec
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x
        for i in range(self.rec):
            x_init = out
            y = self.avgpool(out).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            out = x_init * y.expand_as(x_init)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_features, reduction=16, rec=1):
        super(ChannelAttention, self).__init__()
        self.rec = rec
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.avgpool(x).view(b, c)
        x2 = self.maxpool(x).view(b, c)

        x1 = self.fc(x1)
        x2 = self.fc(x2)
        y = self.sigmoid(x1 + x2)
        y = y.view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cpool = ChannelPool()
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=(7, 7),
                              padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.cpool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class CBAM(nn.Module):
    def __init__(self, in_features, reduction=16):
        super(CBAM, self).__init__()
        self.c_att = ChannelAttention(in_features=in_features,
                                      reduction=reduction)
        self.s_att = SpatialAttention()

    def forward(self, x):
        c_att = self.c_att(x)
        s_att = self.s_att(c_att)
        return s_att


class A2Nework(nn.Module):
    def __init__(self,
                 in_features,
                 cm,
                 cn):
        super(A2Nework, self).__init__()
        self.cm = cm
        self.cn = cn
        self.conv1 = nn.Conv2d(in_channels=in_features,
                               out_channels=cm,
                               kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_features,
                               out_channels=cn,
                               kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=in_features,
                               out_channels=cn,
                               kernel_size=(1, 1))

        self.conv_out = nn.Conv2d(in_channels=cm,
                                  out_channels=in_features,
                                  kernel_size=(1, 1))


    def forward(self, x):
        b, c, h, w = x.shape
        A = self.conv1(x)
        B = self.conv2(x)
        V = self.conv3(x)
        bpool_init = A.view(b, self.cm, -1)
        att_maps = F.softmax(B.view(b, self.cn, -1))
        att_vecs = F.softmax(V.view(b, self.cn, -1))
        gathered = torch.bmm(bpool_init, att_maps.permute(0, 2, 1))
        distributed = gathered.matmul(att_vecs).view(b, self.cm, h, w)
        out = self.conv_out(distributed)
        return x + out


if __name__ == '__main__':
    model = A2Nework(256, 128, 128)
    inp = torch.rand(32, 256, 7, 7)
    out = model(inp)
    print(out.shape)









