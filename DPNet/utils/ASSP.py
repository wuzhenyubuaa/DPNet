import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(ASPPModule,self).__init__()

        self.atrous_conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride=1,padding=padding,dilation=dilation,bias=False)

        self.bn = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU()

        self.initialize()

    def forward(self, x):

        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


    def initialize(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes=256, output_stride=8):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outplanes),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(5*outplanes, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.initialize()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
