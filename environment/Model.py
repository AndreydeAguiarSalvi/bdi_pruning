import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 

'''
    Implements a CNN to MNIST
'''
class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
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
                ('conv', nn.Conv2d(64, 16, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.linear = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(1600, 10)),
                ('function', nn.Softmax())
            ])
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.linear(x.view(-1, 16*10*10))
    
        return x

'''
    Implements a CNN to CIFAR-10
'''
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
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
                ('conv', nn.Conv2d(64, 16, 3, 1)),
                ('function', nn.LeakyReLU())
            ])
        )
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.linear = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(1600, 10)),
                ('function', nn.Softmax())
            ])
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.linear(x.view(-1, 16*10*10))
    
        return x