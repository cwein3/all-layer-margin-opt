import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

__all__ = ['BNWideResNet', 'bn_wideresnet16', 'bn_wideresnet28', 'bn_wideresnet_depth']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, convShortcut=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.convShortcut = convShortcut

    def forward(self, x):

        out = self.bn1(x)
        relu1 = self.relu(out)

        out = self.conv1(relu1)
        out = self.bn2(out)         
        out = self.relu(out)

        out = self.conv2(out)
        

        if self.convShortcut is not None:
            out += self.convShortcut(relu1)
        else:
            out += x
        return out

    def forward_perturb(self, x, deltas):
        bs = x.size(0)
        
        out = self.bn1(x)
        relu1 = self.relu(out)

        out = self.conv1(relu1)
        out = out + torch.norm(out.view(bs, -1), dim=1)[:, None, None, None]*deltas[0]
        out = self.bn2(out)         
        out = self.relu(out)

        out = self.conv2(out)
        out = out + torch.norm(out.view(bs, -1), dim=1)[:, None, None, None]*deltas[1]
        

        if self.convShortcut is not None:
            out += self.convShortcut(relu1)
        else:
            out += x
        return out

    def forward_init_delta(self, x):
        bs = x.size(0)
        
        out = self.bn1(x)
        relu1 = self.relu(out)

        out = self.conv1(relu1)
        delta1 = Variable(torch.zeros(out.size(),dtype=torch.float32,device='cuda:0'),requires_grad=True)
        out = out + torch.norm(out.view(bs, -1), dim=1)[:, None, None, None]*delta1
        out = self.bn2(out)         
        out = self.relu(out)

        out = self.conv2(out)
        delta2 = Variable(torch.zeros(out.size(),dtype=torch.float32,device='cuda:0'),requires_grad=True)
        out = out + torch.norm(out.view(bs, -1), dim=1)[:, None, None, None]*delta2

        if self.convShortcut is not None:
            out += self.convShortcut(relu1)
        else:
            out += x
            
        return out, [delta1, delta2]
    

class NetworkBlock(nn.Module):
    def __init__(self, block, in_planes, out_planes, blocks, start_num, stride=1):
        super(NetworkBlock, self).__init__()
        assert(blocks >= 1)
        self.layers = self._make_layer(block, in_planes, out_planes, blocks, stride)
        self.start_num = start_num # starting layer of the residual block
        self.layer_nn = nn.Sequential(*self.layers)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride):
        convShortcut = nn.Conv2d(
            in_planes, 
            out_planes, 
            kernel_size=1, 
            stride=stride, 
            padding=0, 
            bias=False)

        layers = []
        layers.append(block(in_planes, out_planes, stride, convShortcut))
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))

        return layers

    def forward(self, x):
        return self.layer_nn(x)

    def forward_perturb(self, x, deltas):
        counter = 0
        out= self.layers[0].forward_perturb(x, deltas[0:2])
        for i in range(1, len(self.layers)):
            out = self.layers[i].forward_perturb(out, deltas[2*i:2*(i + 1)])
        return out 

    def forward_init_delta(self, x):
        deltas = []
        out=x
        for i in range(0, len(self.layers)):
            out, curr_deltas = self.layers[i].forward_init_delta(out)
            deltas += curr_deltas
        return out, deltas 

    def perturb_count(self):
        return 2*len(self.layers)

class BNWideResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, width=10):
        super(BNWideResNet, self).__init__()
        self.num_layers = sum(layers)
        start_in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = NetworkBlock(block, start_in_planes, 16*width, layers[0], 0)
        self.layer2 = NetworkBlock(block, 16*width, 32*width, layers[1], layers[0], stride=2)
        self.layer3 = NetworkBlock(block, 32*width, 64*width, layers[2], layers[0]+layers[1], stride=2)
        self.bn2 = nn.BatchNorm2d(64*width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    def forward_perturb(self, x, deltas):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        perturb_count = 0
        perturb_end = perturb_count + self.layer1.perturb_count()
        x = self.layer1.forward_perturb(x, deltas[perturb_count:perturb_end])
        perturb_count= perturb_end
        perturb_end += self.layer2.perturb_count()
        x = self.layer2.forward_perturb(x, deltas[perturb_count:perturb_end])
        perturb_count = perturb_end
        perturb_end += self.layer3.perturb_count()
        x = self.layer3.forward_perturb(x, deltas[perturb_count:perturb_end])
        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_init_delta(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, delta1 = self.layer1.forward_init_delta(x)
        x, delta2 = self.layer2.forward_init_delta(x)
        x, delta3 = self.layer3.forward_init_delta(x)
        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, delta1 + delta2 + delta3

def bn_wideresnet16(**kwargs):
    model = BNWideResNet(BasicBlock, [2, 2, 2], **kwargs)
    return model

def bn_wideresnet28(**kwargs):
    model = BNWideResNet(BasicBlock, [4, 4, 4], **kwargs)
    return model

def bn_wideresnet_depth(depth):
    n = (depth - 4)//6
    model = BNWideResNet(BasicBlock, [n, n, n])
    return model