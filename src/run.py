import os
from tqdm import tqdm
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import (
    datasets,
    transforms,
    models
)
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
    Subset
)

import utils
import models
import coverage
import backdoor.poison_cifar as poison

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset address, model address, image size, label number
scenario_dict = {
    'cifar10': ['./exp_model/vgg16.pt', [3, 32, 32], 10],
}


def get_basic_dataloader(dataset_addr, batch_size=100):
    train_dataset = datasets.CIFAR10(root=dataset_addr, train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]))
    test_dataset = datasets.CIFAR10(root=dataset_addr, train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ]))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, )

    return train_loader, test_loader


def get_biased_dataloader(test_loader, selected_labels, batch_size=100):
    class_num = len(test_loader.dataset.classes)
    assert class_num >= len(selected_labels)

    indices = [idx for idx, target in enumerate(
        test_loader.dataset.targets) if target in selected_labels]

    bias_loader = torch.utils.data.DataLoader(
        Subset(test_loader.dataset, indices),
        batch_size=batch_size, drop_last=False,
        shuffle=True,
    )
    return bias_loader


def test_model(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc



def main(args):
    dataset_addr = './'
    model_addr, image_size, label_num = scenario_dict[args.scenario]

    model = models.VGG('VGG16').to(DEVICE)
    print('model: VGG16')
    state_dict = torch.load(model_addr)
    model.load_state_dict(state_dict)

    train_loader, test_loader = get_basic_dataloader(
        args.scenario, batch_size=args.batch_size)
    class_num = label_num

    # loss, acc=test_model(model, criterion=nn.CrossEntropyLoss(), data_loader=test_loader)
    # print('Test accuracy: %.5f, test loss' % acc, loss)

    rsd = int(datetime.now().timestamp())
    print(f'random seed:{rsd}')
    random.seed(rsd)
    np.random.seed(rsd)
    torch.manual_seed(rsd)
    torch.cuda.manual_seed(rsd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_list = []
    print("fixing setup data...")
    for i, (data, label) in enumerate(tqdm(train_loader)):
        if i >= int(args.setup_size/args.batch_size):
            break
        train_list.append((data, label))

    coverage_dict = []
    start_time = datetime.now()


    input_size = (1, *image_size)
    random_input = torch.randn(input_size).to(DEVICE)
    layer_size_dict = utils.get_layer_inout_sizes(model, random_input)


    model.eval()

    with torch.no_grad():

        cov = coverage.CGRC(model, 8, layer_size_dict,
                            p_value=0.05, chi2_test_threshold=5,
                            is_naive=False, is_plus=False,)

        # Update with Data List
        cov.build(train_list, num=(1000/args.batch_size))
        # cov.load('../temp/cov.pt')
        print(cov.current)
        # cov.save('../temp/cov.pt')

        # Incremental Update
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(DEVICE)
            cov_dict = cov.calculate(data)
            inc = cov.gain(cov_dict)
            if inc is not None:
                cov.update(cov_dict, inc)
            # print(cov.current)
            coverage_dict.append(cov.current)
            # if i >= 50:
            #     break
        print(f"complete testset: {cov.current}")
    # cov_rec=cov.current

    print("Done!")
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    # plot coverage
    plt.plot(coverage_dict)

    plt.xlabel('Batch')
    plt.ylabel('Coverage Rate')
    # set font size
    plt.rcParams.update({'font.size': 20})

    # plt.show()
    # save figure as pdf
    plt.savefig('./CC.png', bbox_inches='tight')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--scenario', '-s',
                        type=str, default='cifar10',
                        choices=['cifar10'],
                        help='scenario choice')
    parser.add_argument('--batch_size', '-b',
                        type=int, default=200,
                        help='batch size')
    parser.add_argument('--setup_size', '-ss',
                        type=int, default=1000,
                        help='setup size')
    args = parser.parse_args()
    print(args)

    main(args)
