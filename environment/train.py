import torch
import argparse
import numpy as np
import pandas as pd
from Model import MNIST_Model, CIFAR_Model
from pruning import create_mask, apply_mask
from utils import create_loaders, create_criterion_optimizer, save_as_csv

def train(n_epochs, train_loader, valid_loader, optimizer, criterion, model, mask):
    running_loss = []
    best_fitness = np.Inf
    for epoch in range(n_epochs): 
        print(f"Performing epoch {epoch}")
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
        print(f"Epoch: {epoch}  Train Loss: {train_loss}  Valid Loss: {valid_loss}")
        running_loss.append(train_loss, valid_loss)

        # Saving last model
        if valid_loss < best_fitness:
            torch.save( model, 'environment/model.pth' )
            torch.save( model, 'environment/mask.pth' )
            best_fitness = valid_loss


def validation(valid_loader, model, mask, criterion):
    print("\tValidating model")
    valid_loss = .0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            apply_mask(model, mask)
            outputs = model(images)
            valid_loss += criterion(outputs, labels)

        valid_loss /= len(valid_loader)

    return valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  
    parser.add_argument('--first_time', action='store_true')
    parser.add_argument('--dataset', type=str, default='CIFAR')
    args = vars(parser.parse_args())

    if args['dataset'] == 'CIFAR':
        model = CIFAR_Model()
        train_loader, valid_loader, _, _ = create_loaders()
    else:
        model = MNIST_Model()
        train_loader, valid_loader, _, _ = create_loaders(which_dataset='MNIST')

    mask = create_mask(model)

    if args['first_time']:
        torch.save(model, 'environment/initial_model.pth')
        torch.save(mask, 'environment/mask.pth')
        save_as_csv(model)
    else:
        model.load_state_dict( torch.load('environment/initial_model.pth') )
        mask.load_state_dict( torch.load('environment/mask.pth') )

    criterion, optimizer = create_criterion_optimizer(model)

    train(args['epochs'], train_loader, valid_loader, optimizer, criterion, model, mask)