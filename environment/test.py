import torch
from Model import Net1, Net2

cifar = torch.Tensor(5, 3, 32, 32)
mnist = torch.Tensor(5, 1, 32, 32)

m1 = Net1()
m2 = Net2()

print(m1(mnist).shape)
print(m2(cifar).shape)