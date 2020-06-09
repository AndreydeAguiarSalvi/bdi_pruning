import torch
from environment.utils import create_loaders, create_criterion_optimizer 

def train(n_epochs, train_loader, valid_loader, optimizer, criterion, model):
    for epoch in range(n_epochs): 
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')


def test():


def main():
