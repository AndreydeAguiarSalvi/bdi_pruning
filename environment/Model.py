import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict 

'''
    Implements a CNN to MNIST
'''
class MNIST_Model(nn.Module):

    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(1, 32, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(32, 64, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv3 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 64, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv4 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 32, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv5 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(32, 16, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.linear1 = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(16*10*10, 512)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.drop = nn.Dropout()
        self.linear2 = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(512, 10)),
                ('function', nn.ReLU())
            ])
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.linear1(x.view(-1, 16*10*10))
        x = self.drop(x)
        x = self.linear2(x)
        
        return x

'''
    Implements a CNN to CIFAR-10
'''
class CIFAR_Model(nn.Module):

    def __init__(self):
        super(CIFAR_Model, self).__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(3, 32, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(32, 64, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv3 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 64, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv4 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 32, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.conv5 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(32, 16, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.linear1 = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(16*10*10, 512)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.drop = nn.Dropout()
        self.linear2 = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(512, 10)),
                ('function', nn.ReLU())
            ])
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.linear1(x.view(-1, 16*10*10))
        x = self.drop(x)
        x = self.linear2(x)

        return x
