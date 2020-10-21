import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_dw(nn.Module):
    def __init__(self, inp, oup):
        super(conv_dw, self).__init__()
        self.dwc = nn.AdaptiveAvgPool2d(1) 
        self.dwc1 = nn.Conv2d(inp, oup, 1, 1, 0, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)         
        self.sig2 = nn.Sigmoid()             
    def forward(self, x): 
        x = self.bn1(self.dwc1(self.dwc(x)))        
        x = self.sig2(x)
        return x

class shortct(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(shortct, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False) 
        
        #<<---- SCM for conv1
        self.cc1 = nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, groups=int(oup/1), bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        #---->>

        self.calib1 = conv_dw(oup, oup) # CCM for conv1
        
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.cc1(out) 
        out = self.bn1(out)
        y = self.calib1(out)        
        out = out*y        
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #<<---- SCM for conv1
        self.cc1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=int(planes/1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #---->>

        self.calib1 = conv_dw(planes, planes) # CCM for conv1

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #<<---- SCM for conv2
        self.cc2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=int(planes/1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #---->>

        self.calib2 = conv_dw(planes, planes) # CCM for conv2


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = shortct(in_planes, self.expansion*planes, stride)
                
        
         

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.cc1(out) 
        out = self.bn1(out)
        y = self.calib1(out)        
        out = out*y        
        out = F.relu(out)
        out = self.conv2(out)
        out = out + self.cc2(out)
        out = self.bn2(out)
        y = self.calib2(out)        
        out = out*y 
        out += self.shortcut(x)
        out = F.relu(out)
        return out


    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cc1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.calib1 = conv_dw(64, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.cc1(out) 
        out = self.bn1(out)
        y = self.calib1(out)        
        out = out*y        
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def Net():
    return ResNet(BasicBlock, [2,2,2,2]) # ResNet-18


print(Net())
