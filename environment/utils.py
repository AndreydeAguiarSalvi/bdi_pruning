import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.torchvision.datasets as D
import torchvision.transforms as transforms
import torch.utils.data.DataLoader as DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def create_loaders(which_dataset='CIFAR'):
    train_loader, valid_loader, test_loader, classes = None, None, None, None

    if which_dataset == 'CIFAR':
        # Data Augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        # Data sets
        train_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        valid_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        test_set = D.CIFAR10(
            root='./data', train=False,
            download=True, transform=test_transform
        )
        # Spliting train in train/validation
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(.3 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_set, batch_size=64,
            shuffle=True, num_workers=4,
            sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid_set, batch_size=64,
            shuffle=True, num_workers=4,
            sampler=valid_sampler
        )
        test_loader = DataLoader(
            test_set, batch_size=64,
            shuffle=False, num_workers=4
        )
        # Classes names
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        # Data Augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Data sets
        train_set = D.MNIST(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        valid_set = D.MNIST(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        test_set = D.MNIST(
            root='./data', train=False,
            download=True, transform=test_transform
        )
        # Spliting train in train/validation
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(.3 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_set, batch_size=64,
            shuffle=True, num_workers=4,
            sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid_set, batch_size=64,
            shuffle=True, num_workers=4,
            sampler=valid_sampler
        )
        test_loader = DataLoader(
            test_set, batch_size=64,
            shuffle=False, num_workers=4
        )
        # Classes names
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') 
    
    return train_loader, valid_loader, test_loader, classes


def plot_images(images, cls_true, label_names, cls_pred=None, save=False, params=None):
    """
        Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    is_CIFAR = '0' in label_names
    if not is_CIFAR: # Dataset is MNIST
        images = images * .5 + .5
        images = images.numpy()
    else: # Dataset is CIFAR-10
        images = images * .3081 + .1307
        images = images.numpy().transpose(1, 2, 0)
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    if save: plt.savefig('wrapping/inference.png', transparent=True )
    else: plt.show()


def create_criterion_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer
