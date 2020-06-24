import torch
import argparse
import pandas as pd
from Model import MNIST_Model, CIFAR_Model
from pruning import create_mask, prune_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR')
    args = vars(parser.parse_args())

    if args['dataset'] == 'CIFAR': model = CIFAR_Model()
    else: model = MNIST_Model()
    
    mask = create_mask(model)
    df = pd.from_csv('wrapping\\pruned_layers.csv', sep=',')

    keys = list( ck.keys() )

    for _ row in df.iterrows():
        layer = int(row[0])
        channel = int(row[1])
        prune_mask( mask[ keys[layer] ], channel )

    torch.save(mask.state_dict("environment\\mask.pth"))