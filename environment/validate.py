import torch
import argparse
import numpy as np
import pandas as pd
from Model import MNIST_Model, CIFAR_Model
from pruning import create_mask, apply_mask
from utils import create_loaders, create_criterion_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR')
    args = vars(parser.parse_args())
    
    if args['dataset'] == 'CIFAR':
        model = CIFAR_Model()
        _, valid_loader, _, _ = create_loaders(is_train=False, is_valid=True, is_test=False)
    else:
        model = MNIST_Model()
        _, valid_loader, _, _ = create_loaders(which_dataset='MNIST', is_train=False, is_valid=True, is_test=False)

    mask = create_mask(model)
    model.load_state_dict( torch.load('environment\\model.pth') )
    mask.load_state_dict( torch.load('environment\\mask.pth') )
    
    criterion, _ = create_criterion_optimizer(model)
    
    valid_loss = .0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            apply_mask(model, mask)
            outputs = model(images)
            valid_loss += criterion(outputs, labels).item()

        valid_loss /= len(valid_loader)

    df = pd.from_csv('wrapping\\pruned_layers.csv', sep=',')
    df[-1, -1] = valid_loss
    df.to_csv('wrapping\\pruned_layers.csv', sep=',')