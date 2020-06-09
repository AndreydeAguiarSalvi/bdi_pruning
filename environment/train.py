import torch
import argparse
import numpy as np
from environment.pruning import create_mask, apply_mask
from environment.utils import create_loaders, create_criterion_optimizer 

def train(n_epochs, train_loader, valid_loader, optimizer, criterion, model, mask):
    running_loss = []
    best_fitness = np.Inf
    for epoch in range(n_epochs): 
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            apply_mask(model, mask)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        valid_loss = validation(valid_loader, model, mask, criterion)
        running_loss.append(train_loss, valid_loss)

        # Save best model


    print('Finished Training')
    if valid_loss < best_fitness:
        torch.save( model, 'environment/model.pth' )
        best_fitness = valid_loss


def validation(valid_loader, model, mask, criterion):
    valid_loss = .0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            apply_mask(model, mask)
            outputs = model(images)
            valid_loss += criterion(outputs, labels)

        valid_loss /= len(valid_loader)
    
    return valid_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  
    
