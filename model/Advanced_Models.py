__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch.nn as nn
import torch.quantization
from model.backbones.main_block import *


class ResUnetPlus(nn.Module):
    def __init__(self,
                 in_features=3,
                 out_features=2,
                 k=1,
                 norm_type='bn',
                 quantization=False
                 ):
        nn.Module.__init__(self)
        self.quantization = quantization

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.ff = torch.nn.quantized.FloatFunctional()

        self.conv1 = nn.Sequential(
                                conv_block(in_features=in_features,
                                           out_features=int(32 * k),
                                           kernel_size=(7, 7),
                                           padding=(3, 3),
                                           norm_type=norm_type,
                                           activation=True),
                               conv_block_noact(in_features=int(32 * k),
                                          out_features=int(32 * k),
                                          kernel_size=(3, 3),
                                          padding=(1, 1),
                                          stride=(2, 2),
                                          norm_type=norm_type,
                                          activation=False)
                                                       )

        self.conv1_skip = conv_block_noact(in_features=in_features,
                                       out_features=int(32 * k),
                                       kernel_size=(3, 3),
                                       padding=(1, 1),
                                        stride=(2, 2),
                                       norm_type=norm_type,
                                       activation=False)

        # self.squeeze_excite1 = CBAM(int(32 * k),
        #                             reduction=int(16 * k))

        self.squeeze_excite1 = SqueezeExciteBlock(int(32 * k),
                                    reduction=int(16 * k))


        self.relu1 = nn.ReLU()

        self.conv2 = nn.Sequential(
                                conv_block(in_features=int(32 * k),
                                           out_features=int(64 * k),
                                           kernel_size=(3, 3),
                                           padding=(1, 1),
                                           norm_type=norm_type,
                                           activation=True),
                               conv_block_noact(in_features=int(64 * k),
                                          out_features=int(64 * k),
                                          kernel_size=(3, 3),
                                          padding=(1, 1),
                                          stride=(2, 2),
                                          norm_type=norm_type,
                                          activation=False)
                                                       )

        self.conv2_skip = conv_block_noact(in_features=int(32 * k),
                                       out_features=int(64 * k),
                                       kernel_size=(3, 3),
                                       padding=(1, 1),
                                     stride=(2, 2),
                                     norm_type=norm_type,
                                       activation=False)

        # self.squeeze_excite2 = CBAM(int(64 * k),
        #                             reduction=int(16 * k))

        self.squeeze_excite2 = SqueezeExciteBlock(int(64 * k),
                                    reduction=int(16 * k))

        self.relu2 = nn.ReLU()


        self.conv3 = nn.Sequential(
                                conv_block(in_features=int(64 * k),
                                           out_features=int(128 * k),
                                           kernel_size=(3, 3),
                                           padding=(1, 1),
                                           norm_type=norm_type,
                                           activation=True),
                               conv_block_noact(in_features=int(128 * k),
                                          out_features=int(128 * k),
                                          kernel_size=(3, 3),
                                          padding=(1, 1),
                                          stride=(2, 2),
                                          norm_type=norm_type,
                                          activation=False)
                                                       )

        self.conv3_skip = conv_block_noact(in_features=int(64 * k),
                                       out_features=int(128 * k),
                                       kernel_size=(3, 3),
                                       padding=(1, 1),
                                     stride=(2, 2),
                                     norm_type=norm_type,
                                       activation=False)

        # self.squeeze_excite3 = CBAM(int(128 * k),
        #                             reduction=int(16 * k))

        self.squeeze_excite3 = SqueezeExciteBlock(int(128 * k),
                                    reduction=int(16 * k))

        self.relu3 = nn.ReLU()


        self.aspp_bridge = ResUASPP(int(128 * k),
                                    int(128 * k),
                                    norm_type=norm_type)

        self.avgpool = Avgpool()

        self.fc = linear_block(in_features=128,
                               out_features=out_features)

        self.sigmoid = nn.Sigmoid()

        self.initialize_weights()

        if self.quantization:
            self.fuse_modules()

    def forward(self, x):
        x = self.quant(x)
        x1 = self.conv1_skip(x)
        x = self.conv1(x)
        x = self.ff.add(x, x1)
        x1 = self.squeeze_excite1(x)
        x = self.ff.add(x, x1)
        x = self.relu1(x)

        x1 = self.conv2_skip(x)
        x = self.conv2(x)
        x = self.ff.add(x, x1)
        x1 = self.squeeze_excite2(x)
        x = self.ff.add(x, x1)
        x = self.relu2(x)

        x1 = self.conv3_skip(x)
        x = self.conv3(x)
        x = self.ff.add(x, x1)
        x1 = self.squeeze_excite3(x)
        x = self.ff.add(x, x1)
        x = self.relu3(x)

        x = self.aspp_bridge(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def fuse_modules(self):
        for m in self.modules():
            if type(m) == conv_block:
                torch.quantization.fuse_modules(m, ['conv', 'norm', 'relu'], inplace=True)
            if type(m) == conv_block_noact:
                torch.quantization.fuse_modules(m, ['conv', 'norm'], inplace=True)


if __name__ == '__main__':

    def test(batchsize):
        in_shape = (batchsize, 3, 120, 160)
        in1 = torch.rand(in_shape).to('cuda')
        model = ResUnetPlus(in_features=3,
                            out_features=2,
                            k=1,
                            norm_type='bn',
                            quantization=True).to('cuda')

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params


    shape, total_params = test(batchsize=32)
    print('Shape : ', shape, '\nTotal params : ', total_params)
