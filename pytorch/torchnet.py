# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:07:13 2020

@author: light_pollusion_team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class wcrn(nn.Module):
    def __init__(self, num_classes=9):
        super(wcrn, self).__init__()

        self.conv1a = nn.Conv2d(103,64,kernel_size=3,stride=1,padding=0,groups=1)
        self.conv1b = nn.Conv2d(103,64,kernel_size=1,stride=1,padding=0,groups=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=3)
        self.maxp2 = nn.MaxPool2d(kernel_size=5)
        
#        self.bn1 = nn.BatchNorm2d(128,eps=0.001,momentum=0.9)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2a = nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0,groups=1)
        self.conv2b = nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0,groups=1)

        self.fc = nn.Linear(128, num_classes)
#        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        
    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        out = self.maxp1(out)
        out1 = self.maxp2(out1)
        
        out = torch.cat((out,out1),1)
        
        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)
        
        out = torch.add(out,out1)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
class resnet99_avg(nn.Module):
    def __init__(self, num_classes=9):
        super(resnet99_avg, self).__init__()
        
        self.conv1a = nn.Conv2d(103,32,kernel_size=3,stride=1,padding=0,groups=1)
        self.conv1b = nn.Conv2d(103,32,kernel_size=3,stride=1,padding=0,groups=1)
        
        self.bn1 = nn.BatchNorm2d(64,eps=0.001,momentum=0.9)
        self.conv2a = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv2b = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,groups=1)
        
        self.bn2 = nn.BatchNorm2d(64,eps=0.001,momentum=0.9)
        self.conv3a = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv3b = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,groups=1)
        
        self.fc = nn.Linear(64, num_classes)
        
        
    def forward(self, x):
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)

        x1 = torch.cat((x1,x2),axis=1)
        x2 = self.bn1(x1)
        x2 = nn.ReLU()(x2)
        x2 = self.conv2a(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.conv2b(x2)
        x1 = torch.add(x1,x2)
        
        x2 = self.bn2(x1)
        x2 = nn.ReLU()(x2)
        x2 = self.conv3a(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.conv3b(x2)
        x1 = torch.add(x1,x2)
        
        x1 = nn.AdaptiveAvgPool2d((1,1))(x1)
        x1 = x1.reshape(x1.size(0), -1)
        
        out = self.fc(x1)
        return out