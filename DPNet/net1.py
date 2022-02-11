#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ASSP import ASPP
from utils.FTP.FPT import Transformer

import sys

sys.path.append('./')



from utils.PyconvResNet import pyconvresnet50, pyconvresnet101, pyconvresnet152, pyconvresnet34, pyconvresnet18
# from utils.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.resnet_official import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.res2net import res2net50_26w_4s, res2net101_26w_4s
from utils.inception_resnetv2 import InceptionResNetV2
from utils.ResNeSt.resnest import resnest50,resnest101,resnest200,resnest269
from utils.densenet import DenseNet
# from utils.swin_transformer_segmentor.swin_tranformer_segmentation import SwinTransformer
from utils.ASFF import ASFF
from utils.PPM import PPM

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.LeakyReLU) or isinstance(m, nn.Softmax):
            pass 
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/userhome/wuzhenyu/mywork/work5/to_pcl_server/res/resnet50-19c8e357.pth'), strict=False)


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)

class SKConv(nn.Module):
    def __init__(self, features, WH=None, M=2, G=1, r=1, stride=1, L=32):
        """ Constructor SKNet
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        # self.convs = nn.ModuleList([])
        # for i in range(M):
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
        #         nn.BatchNorm2d(features),
        #         nn.ReLU(inplace=False)
        #     ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
        self.fc1 = nn.Linear(d, features)
        self.fc2 = nn.Linear(d, features)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x0, x1):
        # for i, conv in enumerate(self.convs):
        #     fea = conv(x).unsqueeze_(dim=1)
        #     if i == 0:
        #         feas = fea
        #     else:
        #         feas = torch.cat([feas, fea], dim=1)

        x0 = x0.unsqueeze(dim=1)
        x1 = x1.unsqueeze(dim=1)

        feas = torch.cat([x0, x1], dim=1)

        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
                
        vector1 = self.fc1(fea_z).unsqueeze(dim=1)
        vector2 = self.fc2(fea_z).unsqueeze(dim=1)
        attention_vectors = torch.cat([vector1, vector2], dim=1)
        
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v
    
    def initialize(self):
        weight_init(self)

class SCFM(nn.Module):
    '''
    selectively  feature  fusion, author: wuzhenyu, date: 2021/1/15
    '''
    def __init__(self):
        super(SCFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

        self.sf1 = SKConv(features=64)
        self.sf2 = SKConv(features=64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v

        # out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        # out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)



        out4h = self.sf1(out1h, F.relu(self.bn3h(self.conv3h(fuse )), inplace=True))

        # out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        # out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)

        out4v = self.sf2(out1v, F.relu(self.bn3v(self.conv3v(fuse )), inplace=True))

        return out4h, out4v

    def initialize(self):
        weight_init(self)        
        
        
        
        

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v)
            out3h, out3v = self.cfm34(out3h+refine3, out4v)
            out2h, pred  = self.cfm23(out2h+refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred  = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)
        
        
# author: wuzhenyu, date: 2021/1/12
class BiDecoder(nn.Module):
    def __init__(self):
        super(BiDecoder, self).__init__()

        # top to bottom: t2b
        self.cfm45_t2b = CFM()
        self.cfm34_t2b = CFM()
        self.cfm23_t2b = CFM()

        # bottom to top: b2t
        self.cfm45_b2t = CFM()
        self.cfm34_b2t = CFM()
        self.cfm23_b2t = CFM()

#         # top to bottom: tt2b
#         self.cfm45_tt2b = CFM()
#         self.cfm34_tt2b = CFM()
#         self.cfm23_tt2b = CFM()

        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        # top to bottom
        out4h_1, out4v = self.cfm45_t2b(out4h, out5v)
        out3h_1, out3v = self.cfm34_t2b(out3h, out4v)
        out2h_1, pred = self.cfm23_t2b(out2h, out3v)

        # bottom to top
        out2h_2, out2v = self.cfm23_b2t(out2h_1 + out2h, pred)
        out3h_2, out3v = self.cfm34_b2t(out3h_1 + out3h, out2v)
        out4h_2, out4v = self.cfm45_b2t(out4h_1 + out4h, out3v)
        
        if out5v.size()[2:] != out4v.size()[2:]:
            out4v = F.interpolate(out4v, size=out5v.size()[2:], mode='bilinear')

        out5v = F.relu(self.bn(self.conv(out5v + out4v)), inplace=True)

#         # top to bottom
#         out4h, out4v = self.cfm45_tt2b(out4h_2, out5v)
#         out3h, out3v = self.cfm34_tt2b(out3h_2, out4v)
#         out2h, pred = self.cfm23_tt2b(out2h_2, out3v)

        return out2h_2, out3h_2, out4h_2, out5v, pred

    def initialize(self):
        weight_init(self)
        

        
    


class DPNet(nn.Module):
    def __init__(self, cfg):
        super(DPNet, self).__init__()
        self.cfg      = cfg
        self.bkbone = pyconvresnet50()
#         self.bkbone   = resnet50()
#         self.bkbone = ResNet()
#         self.bkbone = res2net101_26w_4s()
#         self.bkbone = InceptionResNetV2()
#         self.bkbone = resnest50()
#         self.bkbone = resnest200()
#         self.bkbone = DenseNet(32, (6, 12, 24, 16), 64) # densenet121, 
#         self.bkbone = DenseNet(32, (6, 12, 32, 32), 64) # densenet169, 
#         self.bkbone = DenseNet(32, (6, 12, 48, 32), 64) # densenet201, 
#         self.bkbone = SwinTransformer()
    
#         self.aspp5 = ASPP(inplanes=2048,outplanes=64)
#         self.aspp4= ASPP(inplanes=1024, outplanes=64)
#         self.aspp3= ASPP(inplanes=512, outplanes=64)
#         self.aspp2 = ASPP(inplanes=256, outplanes=64)
        
#         self.ppm5 = PPM(2048, 64)
#         self.ppm4 = PPM(1024, 64)
#         self.ppm3 = PPM(512, 64)
#         self.ppm2 = PPM(256, 64)
        
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder1 = BiDecoder()
        self.decoder2 = BiDecoder()
        
#         self.asff1 = ASFF(level=2)
#         self.asff2 = ASFF(level=2)
#         self.asff3 = ASFF(level=1)
#         self.asff4 = ASFF(level=1)
#         self.asff5 = ASFF(level=0)
        
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        
        self.initialize()

    def forward(self, x, shape=None):
        out2h, out3h, out4h, out5v        = self.bkbone(x)
        
#         out2h, out3h, out4h, out5v = self.aspp2(out2h), self.aspp3(out3h), self.aspp4(out4h), self.aspp5(out5v)
        
        out2h, out3h, out4h, out5v        = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        
#         out2h, out3h, out4h, out5v        = self.ppm2(out2h), self.ppm3(out3h), self.ppm4(out4h), self.ppm5(out5v)
        

        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, pred1)
        
#         print(pred2.shape, out5v.shape, out4h.shape, out3h.shape, out2h.shape)
        
#         pred2 = self.asff1(out4h, out3h, pred2)
        
#         out2h = self.asff2(out4h, out3h, out2h)
        
#         out3h = self.asff3( out4h, out3h, out2h)
        
#         out4h = self.asff4(out5v, out4h, out3h)
        
#         out5v = self.asff5(out5v, out4h, out3h)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        
        
        return pred1, pred2, out2h, out3h, out4h, out5h


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
