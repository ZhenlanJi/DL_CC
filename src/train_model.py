import os
from pickletools import optimize
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

import models


def get_dataloaders(dataset_name, validation_fraction, batch_size, seed):
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='../temp', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize(
                                            (32, 32)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        test_dataset = datasets.MNIST(root='../temp', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.Resize(
                                              (32, 32)),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='../temp', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize(
                                                 (32, 32)),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
        test_dataset = datasets.CIFAR10(root='../temp', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(
                                                (32, 32)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif dataset_name == 'ImageNet':
        train_dataset = datasets.ImageFolder(root='../temp/train', 
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.Resize(
                                                     (32, 32)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ]))
        test_dataset = datasets.ImageFolder(root='../temp/test', 
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(
                                                    (32, 32)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                            ]))
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset_name))

    if validation_fraction != 0:
        train_size = len(train_dataset)
        val_num = int(validation_fraction * train_size)
        train_subset, val_subset = random_split(
            train_dataset, [train_size-val_num, val_num], generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(
            dataset=train_subset, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(
            dataset=val_subset, shuffle=False, batch_size=batch_size)

    else:
        train_loader = DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = None

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # if args.dataset == 'cifar10':
        #     loss = loss + (torch.norm(model.fc1.weight, p=2) + \
        #         torch.norm(model.fc2.weight, p=2)) * 0.01
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
   # argparse
    parser = argparse.ArgumentParser(description='PyTorch training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name (mnist, cifar10 or ImageNet)')
    parser.add_argument('--batch-size',
                        type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size',
                        type=int, default=2000,
                        help='input batch size for testing (default: 2000)')
    parser.add_argument('--epochs',
                        type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr',
                        type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval',
                        type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validation-fraction', '-vf',
                        type=float, default=0.2,
                        help='fraction of training data to use as validation')
    args = parser.parse_args()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load data
    train_loader, test_loader, val_loader = get_dataloaders(
        args.dataset, args.validation_fraction, args.batch_size, args.seed)

    # create model
    if args.dataset == 'mnist':
        model = models.LeNet_5().to(DEVICE)
    elif args.dataset == 'cifar10':
        model = models.VGG_16().to(DEVICE)
    elif args.dataset == 'ImageNet':
        model = models.VGG_16().to(DEVICE)
    else:
        raise ValueError('Invalid dataset name')
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=1, gamma=0.8)
    for epoch in range(1, args.epochs + 1):
        train(model, DEVICE, train_loader,
              optimizer, epoch, args.log_interval)
        if val_loader is not None:
            validate(model, DEVICE, val_loader)
        # scheduler.step()
    print("=================================\n")
    print("Evaluating on test set...")
    evaluate(model, DEVICE, test_loader)
    print("=================================\n")
    print("Saving model...")
    torch.save(model.state_dict(), "../output/model/"+args.dataset+".pt")
    print("=================================\n")
    print("Done!")
