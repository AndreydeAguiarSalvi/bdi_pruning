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
    scheduler = create_scheduler(args, optimizer)
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            if i % 2000 == 0: print(f"Step: {i}")
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

        scheduler.step()

        train_loss /= len(train_loader)
        valid_loss = validation(valid_loader, model, mask, criterion)
        print(f"Epoch: {epoch}  Train Loss: {train_loss}  Valid Loss: {valid_loss}")
        running_loss.append(epoch, train_loss, valid_loss)

        # Saving last model
        if valid_loss < best_fitness:
            torch.save( model.state_dict(), 'environment\\model.pth' )
            torch.save( mask.state_dict(), 'environment\\mask.pth' )
            best_fitness = valid_loss

    df = pd.DataFrame(running_loss, columns=['epoch', 'train_loss', 'valid_loss'])
    df.to_csv('environment\\results.csv', sep=',')
    
    return best_fitness


def validation(valid_loader, model, mask, criterion):
    print("\tValidating model")
    valid_loss = .0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            apply_mask(model, mask)
            outputs = model(images)
            valid_loss += criterion(outputs, labels).item()

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
        train_loader, valid_loader, _, _ = create_loaders(is_train=True, is_valid=True, is_test=False)
    else:
        model = MNIST_Model()
        train_loader, valid_loader, _, _ = create_loaders(which_dataset='MNIST', is_train=True, is_valid=True, is_test=False)

    mask = create_mask(model)

    if args['first_time']:
        torch.save(model.state_dict(), 'environment\\initial_model.pth')
        torch.save(mask.state_dict(), 'environment\\initial_mask.pth')
        save_as_csv(model)
    else:
        model.load_state_dict( torch.load('environment\\model.pth') )
        mask.load_state_dict( torch.load('environment\\mask.pth') )

    criterion, optimizer = create_criterion_optimizer(model)

    value = train(args['epochs'], train_loader, valid_loader, optimizer, criterion, model, mask)
    
    f = open('wrapping\\result.txt', 'w')
    print(value, file=f, end='')
    f.close()