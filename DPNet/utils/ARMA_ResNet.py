import torch
import torch.nn as nn
import torch.nn.functional as F
from .ARMA import ARMA2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, arma, w_ksz, a_ksz, stride=1, init=0):
        super(BasicBlock, self).__init__()
        if arma:
            self.conv1 = ARMA2d(in_planes, planes, w_stride=stride, w_kernel_size=w_ksz, w_padding=w_ksz // 2,
                                 a_kernel_size=a_ksz, a_padding=a_ksz // 2)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=w_ksz, stride=stride, padding=w_ksz // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if arma:
            self.conv2 = ARMA2d(planes, planes, w_kernel_size=w_ksz, w_padding=w_ksz // 2,
                                 a_kernel_size=a_ksz, a_padding=a_ksz // 2)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=w_ksz, stride=1, padding=w_ksz // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if arma:
                self.shortcut = nn.Sequential(
                    ARMA2d(in_planes, self.expansion * planes, w_kernel_size=1, w_stride=stride, w_padding=0,
                            a_kernel_size=a_ksz, a_padding=a_ksz // 2),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, arma, w_ksz, a_ksz, stride=1, init=0):
        super(Bottleneck, self).__init__()
        if arma:
            self.conv1 = ARMA2d(in_planes, planes, w_kernel_size=1, w_padding=0,
                                a_kernel_size=a_ksz, a_padding=a_ksz)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if arma:
            self.conv2 = ARMA2d(planes, planes, w_stride=stride, w_kernel_size=w_ksz, w_padding=w_ksz // 2,
                                 a_kernel_size=a_ksz, a_padding=a_ksz // 2)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=w_ksz, stride=stride, padding=w_ksz // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if arma:
            self.conv3 = ARMA2d(planes, self.expansion * planes, w_kernel_size=1, w_padding=0)
        else:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if arma:
                self.shortcut = nn.Sequential(
                    ARMA2d(in_planes, self.expansion * planes, w_kernel_size=1, w_stride=stride, w_padding=0),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, arma, w_ksz, a_ksz):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if arma:
            self.conv1 = ARMA2d(3, 64, w_kernel_size=w_ksz, w_padding=w_ksz // 2,
                               a_kernel_size=a_ksz, a_padding=a_ksz // 2)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=w_ksz, stride=1, padding=w_ksz // 2, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], arma, w_ksz, a_ksz,
                                       stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], arma, w_ksz, a_ksz,
                                       stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], arma, w_ksz, a_ksz,
                                       stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], arma, w_ksz, a_ksz,
                                       stride=2)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, arma, w_ksz, a_ksz, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, arma, w_ksz, a_ksz, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # out = self.softmax(out)
        return out2, out3, out4, out5
    
    def initialize(self):
        self.load_state_dict(torch.load('/userhome/wuzhenyu/mywork/work5/to_pcl_server/res/resnet50-19c8e357.pth'), strict=False)


def ResNet_(model_arch="ResNet50", arma=True, rf_init=0,
            w_kernel_size=3, a_kernel_size=3):

    block, num_blocks = {"ResNet18": (BasicBlock, [2, 2, 2, 2]),
                         "ResNet34": (BasicBlock, [3, 4, 6, 3]),
                         "ResNet50": (Bottleneck, [3, 4, 6, 3]),
                         "ResNet101": (Bottleneck, [3, 4, 23, 3]),
                         "ResNet152": (Bottleneck, [3, 8, 36, 3])}[model_arch]

    return ResNet(block, num_blocks, arma, w_kernel_size, a_kernel_size)


if __name__ == '__main__':

    input = torch.rand(2,3,128,128).cuda()

    resnet = ResNet_(model_arch='ResNet50').cuda()

    out = resnet(input)

