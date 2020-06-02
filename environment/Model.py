import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 

'''
    Implements a CNN to MNIST
'''
class Net1(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(1, 32, 3, 1)),
                ('function', F.leaky_relu)
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(32, 64, 3, 1)),
                ('function', F.leaky_relu)
            ])
        )
        self.conv3 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 64, 3, 1)),
                ('function', F.leaky_relu)
            ])
        )
        self.conv4 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(64, 16, 3, 1)),
                ('function', F.leaky_relu)
            ])
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.linear = nn.Sequential(
            OrderedDict([
                ('mlp', nn.Linear(144, 10)),
                ('function', F.softmax)
            ])
        )