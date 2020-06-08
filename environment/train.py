import torch
import torch.torchvision.datasets as D
import torchvision.transforms as transforms
import torch.utils.data.DataLoader as DataLoader

ARGS = 'CIFAR'

if ARGS == 'CIFAR':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = D.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = D.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    transform = transform.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform,
        batch_size=args.batch_size, shuffle=True, **kwargs))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform)),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
