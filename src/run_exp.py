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
import torchattacks

import utils
import models
import coverage
import train_model
import backdoor.poison_cifar as poison

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset address, model address, image size, label number
scenario_dict = {
    'mnist': ['./temp', './output/model/mnist_lenet_5.pt', [1, 32, 32], 10],
    'cifar10': ['./temp', './output/model/vgg16.pt', [3, 32, 32], 10],
    'backdoor': ['./temp', './output/model/badnet_vgg.pt', [3, 32, 32], 10],
    # 'imagenet': ['/home/zjiae/Dataset/ImageNet/ILSVRC/Data/CLS-LOC', None, [3, 224, 224], 1000],
    'imagenet': ['/home/zjiae/Project/Causal-Coverage/temp', None, [3, 224, 224], 20],
}


def get_basic_dataloader(scenario, batch_size=100):
    dataset_addr = scenario_dict[scenario][0]
    if scenario == 'cifar10' or scenario == 'backdoor':
        train_dataset = datasets.CIFAR10(root=dataset_addr, train=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                         ]))
        test_dataset = datasets.CIFAR10(root=dataset_addr, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ]))
    elif scenario == 'mnist':
        train_dataset = datasets.MNIST(root=dataset_addr, train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(
                                               (32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        test_dataset = datasets.MNIST(root=dataset_addr, train=False,
                                      transform=transforms.Compose([
                                          transforms.Resize(
                                              (32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
    elif scenario == 'imagenet':
        # train_dataset = datasets.ImageFolder(root=os.path.join('/home/zjiae/Dataset/ImageNet/ILSVRC/Data/CLS-LOC', 'train'),
        #                                      transform=transforms.Compose([
        #                                          #  transforms.RandomResizedCrop(
        #                                          #      224),
        #                                          #  transforms.RandomHorizontalFlip(),
        #                                          transforms.Resize(256),
        #                                          transforms.CenterCrop(224),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize(
        #                                              (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                      ]))
        # test_dataset = datasets.ImageFolder(root=os.path.join(dataset_addr, 'val'),
        #                                     transform=transforms.Compose([
        #                                         transforms.Resize(256),
        #                                         transforms.CenterCrop(224),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize(
        #                                             (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                     ]))
        train_dataset = torch.load(os.path.join(
            dataset_addr, 'imgnet_20_train.pt'))
        test_dataset = torch.load(os.path.join(
            dataset_addr, 'imgnet_20_test.pt'))
        # print(len(test_dataset))
    else:
        raise ValueError('Scenario not supported')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, )

    return train_loader, test_loader


def get_backdoor_dataloader(trigger_info_addr, data_loader, batch_size=100):
    trigger_info = torch.load(trigger_info_addr)
    backdoor_set = poison.add_predefined_trigger_cifar(
        data_set=data_loader.dataset, trigger_info=trigger_info)
    backdoor_loader = DataLoader(
        dataset=backdoor_set, batch_size=batch_size, shuffle=True, )

    return backdoor_loader


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


def noise_task(image_size, model, train_loader, test_loader, class_num, args):
    input_size = (1, *image_size)
    random_input = torch.randn(input_size).to(DEVICE)
    # layer_size_dict = utils.get_layer_inout_sizes(model, random_input)
    layer_size_dict = utils.get_layer_output_sizes(model, random_input)

    noise_dict = {}
    model.eval()
    with torch.no_grad():
        for coef in [0, 1, 100, 10000]:
            # cov = coverage.CGRC(model, 8, layer_size_dict,)
            # cov = coverage.NBC(model, 10, layer_size_dict,)
            cov = coverage.SNAC(model, 10, layer_size_dict,)
            # cov = coverage.KMNC(model, 1000, layer_size_dict,)
            # cov = coverage.TKNC(model, 10, layer_size_dict,)
            # cov = coverage.TKNP(model, 10, layer_size_dict,)
            # cov = coverage.CC(model, 2000, layer_size_dict,)

            cov.build(train_loader, num=int(args.setup_size/args.batch_size))
            print(f'finish init, current: {cov.current}')

            for i, (data, label) in enumerate(tqdm(test_loader)):
                noise = (torch.rand_like(data)-0.5) * 2 * coef
                data = data + noise
                data = data.to(DEVICE)
                cov_dict = cov.calculate(data)
                inc = cov.gain(cov_dict)
                if inc is not None:
                    cov.update(cov_dict, inc)
            print(f"complete testset: {cov.current}")
            print('----------------------------------------------------\n')
            noise_dict[coef] = cov.current
    return noise_dict


def rq1_task(image_size, model, train_loader, test_loader, class_num, args):
    input_size = (1, *image_size)
    random_input = torch.randn(input_size).to(DEVICE)
    layer_size_dict = utils.get_layer_inout_sizes(model, random_input)
    # layer_size_dict = utils.get_layer_output_sizes(model, random_input)

    model.eval()
    with torch.no_grad():
        shuffle_labels = list(range(class_num))
        random.shuffle(shuffle_labels)
        # print(f"shuffle_labels: {shuffle_labels}")

        bias_result_dict = {}
        is_saved = False
        for ratio in [0.3,0.5,1.0]:
            selected_labels = shuffle_labels[:int(ratio*class_num)]
            biased_loader = get_biased_dataloader(
                test_loader, selected_labels, batch_size=args.batch_size)

            if args.mini:
                if args.scenario == 'mnist':
                    print(f"CGRC mini for MNIST")
                    sub_models = [
                        nn.Sequential(
                            nn.MaxPool2d(2), list(model.children())[1],
                            nn.MaxPool2d(2), nn.Flatten(), list(model.children())[2]),
                        nn.Sequential(
                            nn.ReLU(inplace=True), list(model.children())[3],
                            nn.ReLU(inplace=True), list(model.children())[4])
                    ]
                    cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                             ['Conv2d-1', 'Linear-1', 'Linear-3'])

                elif args.scenario == 'cifar10':
                    # sub_models = [
                    #     nn.Sequential(*list(model.children())[0][38:41]),
                    #     nn.Sequential(*list(model.children())[0][41:],
                    #                     nn.Flatten(), *list(model.children())[1:])
                    # ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                             ['Conv2d-12', 'Conv2d-13', 'Linear-1'])
                    # print(f"CGRC mini of last 3 layers for CIFAR10")

                    # sub_models = [
                    #     nn.Sequential(*list(model.children())[0][21:25]),
                    #     nn.Sequential(*list(model.children())[0][25:28])
                    # ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                          ['Conv2d-7', 'Conv2d-8', 'Conv2d-9'])
                    # print(f"CGRC mini of middle 3 layers for CIFAR10")

                    # sub_models = [
                    #     nn.Sequential(*list(model.children())[0][1:4]),
                    #     nn.Sequential(*list(model.children())[0][4:8])
                    # ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                          ['Conv2d-1', 'Conv2d-2', 'Conv2d-3'])
                    # print(f"CGRC mini of first 3 layers for CIFAR10")

                    sub_models = [
                        nn.Sequential(*list(model.children())[0][1:25]),
                        nn.Sequential(*list(model.children())[0][25:],
                                      nn.Flatten(), *list(model.children())[1:])
                    ]
                    cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                             ['Conv2d-1', 'Conv2d-8', 'Linear-1'])
                    print(f"CGRC mini of optimal 3 layers for CIFAR10")
                elif args.scenario == 'imagenet':
                    sub_models = [
                        nn.Sequential(*list(model.children())[1:6]),
                        nn.Sequential(
                            *list(model.children())[6:-1],
                            nn.Flatten(), list(model.children())[-1]), ]
                    cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                             ['Conv2d-1', 'Bottleneck-7', 'Linear-1'])
                    print(f"CGRC mini of optimal 3 layers for ResNet50")

                    # sub_models = [
                    #     nn.Sequential(*list(model.children())[1:3],list(model.children())[4][0]),
                    #     nn.Sequential(list(model.children())[4][1]), ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                          ['Conv2d-1', 'Bottleneck-1', 'Bottleneck-2'])
                    # print(f"CGRC mini of first 3 layers for ResNet50")

                    # sub_models = [
                    #     nn.Sequential(list(model.children())[6][0]),
                    #     nn.Sequential(list(model.children())[6][1]), ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                          ['Bottleneck-7', 'Bottleneck-8', 'Bottleneck-9'])
                    # print(f"CGRC mini of middle 3 layers for ResNet50")

                    # sub_models = [
                    #     nn.Sequential(list(model.children())[7][-1]),
                    #     nn.Sequential(list(model.children())[8], nn.Flatten(), list(model.children())[-1]), ]
                    # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                    #                          ['Bottleneck-15', 'Bottleneck-16', 'Linear-1'])
                    # print(f"CGRC mini of last 3 layers for ResNet50")
                else:
                    raise ValueError('Scenario not supported')
            else:
                cov = coverage.CGRC(model, 8, layer_size_dict,)
                # cov=coverage.NC(model, 0, layer_size_dict,)
                # cov = coverage.NCS(model, 0.75, layer_size_dict,)
                # cov = coverage.KMNC(model, 1000, layer_size_dict,)
                # cov = coverage.NBC(model, 10, layer_size_dict,)
                # cov = coverage.SNAC(model, 10, layer_size_dict,)
                # cov = coverage.TKNC(model, 10, layer_size_dict,)
                # cov = coverage.TKNP(model, 10, layer_size_dict,)
                # cov = coverage.CC(model, 10, layer_size_dict,)
                # cov = coverage.LSA(model, 1, 1e-5, 10, layer_size_dict,)
                # cov = coverage.DSA(model, 0.1, 0, 10, layer_size_dict,)
                # cov = coverage.MDSA(model, 10, 0, 10, layer_size_dict,)

            if is_saved:
                cov.load('./temp/cov.pt')
            else:
                cov.build(train_loader, num=int(
                    args.setup_size/args.batch_size))
                cov.save('./temp/cov.pt')
                # is_saved = True
            print(f'finish init, current: {cov.current}')

            for i, (data, label) in enumerate(tqdm(biased_loader)):
                if i >= len(test_loader)*0.3:
                    break
                data = data.to(DEVICE)
                cov_dict = cov.calculate(data)
                inc = cov.gain(cov_dict)
                if inc is not None:
                    cov.update(cov_dict, inc)
            print(f"complete testset: {cov.current}")
            print('----------------------------------------------------\n')
            bias_result_dict[ratio] = cov.current
    return bias_result_dict


def rq2_task(image_size, model, train_loader, test_loader, class_num, args):
    input_size = (1, *image_size)
    random_input = torch.randn(input_size).to(DEVICE)
    # layer_size_dict = utils.get_layer_inout_sizes(model, random_input)
    layer_size_dict = utils.get_layer_output_sizes(model, random_input)

    res_dict = {}
    model.eval()
    with torch.no_grad():
        if args.mini:
            if args.scenario == 'mnist':
                print(f"CGRC mini for MNIST")
                sub_models = [
                    nn.Sequential(
                        nn.MaxPool2d(2), list(model.children())[1],
                        nn.MaxPool2d(2), nn.Flatten(), list(model.children())[2]),
                    nn.Sequential(
                        nn.ReLU(inplace=True), list(model.children())[3],
                        nn.ReLU(inplace=True), list(model.children())[4])
                ]
                cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                         ['Conv2d-1', 'Linear-1', 'Linear-3'])

            elif args.scenario == 'cifar10' or args.scenario == 'backdoor':
                # sub_models = [
                #     nn.Sequential(*list(model.children())[0][38:41]),
                #     nn.Sequential(*list(model.children())[0][41:],
                #                   nn.Flatten(), *list(model.children())[1:])
                # ]
                # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                #                          ['Conv2d-12', 'Conv2d-13', 'Linear-1'])
                # print(f"CGRC mini of last 3 layers for CIFAR10")

                # sub_models = [
                #     nn.Sequential(*list(model.children())[0][21:25]),
                #     nn.Sequential(*list(model.children())[0][25:28])
                # ]
                # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                #                             ['Conv2d-7', 'Conv2d-8', 'Conv2d-9'])
                # print(f"CGRC mini of middle 3 layers for CIFAR10")

                # sub_models = [
                #     nn.Sequential(*list(model.children())[0][1:4]),
                #     nn.Sequential(*list(model.children())[0][4:8])
                # ]
                # cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                #                          ['Conv2d-1', 'Conv2d-2', 'Conv2d-3'])
                # print(f"CGRC mini of first 3 layers for CIFAR10")

                sub_models = [
                    nn.Sequential(*list(model.children())[0][1:25]),
                    nn.Sequential(*list(model.children())[0][25:],
                                  nn.Flatten(), *list(model.children())[1:])
                ]
                cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                         ['Conv2d-1', 'Conv2d-8', 'Linear-1'])
                print(f"CGRC mini of optimal 3 layers for CIFAR10")

            elif args.scenario == 'imagenet':
                print(f"CGRC mini for ImageNet")
                sub_models = [
                    nn.Sequential(*list(model.children())[1:7]),
                    nn.Sequential(
                        *list(model.children())[7:-1],
                        nn.Flatten(), list(model.children())[-1]), ]
                cov = coverage.CGRC_mini(model, 8, layer_size_dict, sub_models,
                                         ['Conv2d-1', 'Bottleneck-13', 'Linear-1'])
            else:
                raise ValueError('Scenario not supported')
        else:
            # cov = coverage.CGRC(model, 8, layer_size_dict)
            # cov=coverage.NC(model, 0, layer_size_dict,)
            # cov = coverage.NCS(model, 0.25, layer_size_dict,)
            # cov = coverage.KMNC(model, 1000, layer_size_dict,)
            # cov = coverage.NBC(model, 10, layer_size_dict,)
            # cov = coverage.SNAC(model, 10, layer_size_dict,)
            # cov = coverage.TKNC(model, 10, layer_size_dict,)
            # cov = coverage.TKNP(model, 5, layer_size_dict,)
            # cov = coverage.CC(model, 20, layer_size_dict,)
            cov = coverage.LSA(model, 1, 1e-5, 10, layer_size_dict,)
            # cov = coverage.DSA(model, 0.01, 0, 10, layer_size_dict,)
            # cov = coverage.MDSA(model, 10, 0, 10, layer_size_dict,)

        cov.build(train_loader, num=int(args.setup_size/args.batch_size))
        print(f'finish init, current: {cov.current}')

        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(DEVICE)
            cov_dict = cov.calculate(data, label)
            inc = cov.gain(cov_dict)
            if inc is not None:
                cov.update(cov_dict, inc)
        init_coverage = cov.current
        res_dict['init'] = init_coverage
        print(f"complete testset for init: {cov.current}")

        if args.rq2_task == 'poison':
            backdoor_dataloader = get_backdoor_dataloader(
                args.trigger_info, test_loader, args.batch_size)
            for i, (data, label) in enumerate(tqdm(backdoor_dataloader)):
                data = data.to(DEVICE)
                cov_dict = cov.calculate(data, label)
                inc = cov.gain(cov_dict)
                if inc is not None:
                    cov.update(cov_dict, inc)
        else:
            for i, (data, label) in enumerate(tqdm(test_loader)):
                if args.rq2_task == 'randn':
                    data = torch.randn_like(data).to(DEVICE)
                elif args.rq2_task == 'PGD':
                    with torch.enable_grad():
                        atk = torchattacks.PGD(model)
                        atk.set_return_type(type='float')
                        data = atk(data, label).to(DEVICE)
                elif args.rq2_task == 'FGSM':
                    with torch.enable_grad():
                        # atk = torchattacks.RFGSM(
                        #     model, eps=32/255, alpha=16/255, steps=10)
                        atk=torchattacks.FGSM(model, eps=0.3)
                        atk.set_return_type(type='float')
                        data = atk(data, label).to(DEVICE)
                cov_dict = cov.calculate(data, label)
                inc = cov.gain(cov_dict)
                if inc is not None:
                    cov.update(cov_dict, inc)
        print(f"complete incremental: {cov.current}")
        inc_ratio = (cov.current-init_coverage)/init_coverage
        res_dict['inc'] = inc_ratio
        print(
            f"now: {cov.current}, incremental ratio of {args.rq2_task}: {inc_ratio*100}%")
        print('----------------------------------------------------\n')
        return res_dict


def main(args):
    dataset_addr, model_addr, image_size, label_num = scenario_dict[args.scenario]

    if args.scenario == 'imagenet':
        model = torchvision.models.resnet50(pretrained=True).to(DEVICE)
        print('model: resnet50')
    else:
        if args.scenario == 'mnist':
            model = models.LeNet_5().to(DEVICE)
            print('model: LeNet_5')
        elif args.scenario == 'cifar10':
            model = models.VGG('VGG16').to(DEVICE)
            print('model: VGG16')
        elif args.scenario == 'backdoor':
            model = models.VGG('VGG16').to(DEVICE)
            print('model: VGG16_badnet')
        else:
            raise ValueError('Unknown scenario')

        state_dict = torch.load(model_addr)
        model.load_state_dict(state_dict)

    train_loader, test_loader = get_basic_dataloader(
        args.scenario, batch_size=args.batch_size)
    # class_num = len(train_loader.dataset.classes)
    class_num = label_num

    # loss, acc=test_model(model, criterion=nn.CrossEntropyLoss(), data_loader=test_loader)
    # print('Test accuracy: %.5f, test loss' % acc, loss)

    results = []
    for i in range(args.repeat):
        train_list = []
        print("fixing setup data...")
        for i, (data, label) in enumerate(tqdm(train_loader)):
            if i >= int(args.setup_size/args.batch_size):
                break
            train_list.append((data, label))

        rsd = int(datetime.now().timestamp())
        print(f'random seed:{rsd}')
        random.seed(rsd)
        np.random.seed(rsd)
        torch.manual_seed(rsd)
        torch.cuda.manual_seed(rsd)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.task == 'rq1':
            bias_result_dict = rq1_task(
                image_size, model, train_list, test_loader, class_num, args)
            results.append(bias_result_dict)
        elif args.task == 'noise':
            noise_result_dict = noise_task(
                image_size, model, train_list, test_loader, class_num, args)
            results.append(noise_result_dict)
        elif args.task == 'rq2':
            inc_res = rq2_task(
                image_size, model, train_list, test_loader, class_num, args)
            results.append(inc_res)
        else:
            raise ValueError('Unknown task')

    # print results
    if args.task == 'rq1':
        for ratio in [0.3, 0.5, 1.0]:
            cov_mean_list = []
            for result in results:
                cov_mean_list.append(result[ratio])
            print('==============================================================')
            print(f'{ratio} : {np.mean(cov_mean_list):.5f}')
    elif args.task == 'noise':
        for coeff in [0, 1, 100, 10000]:
            cov_mean_list = []
            for result in results:
                cov_mean_list.append(result[coeff])
            print('==============================================================')
            print(f'coeff {coeff} : {np.mean(cov_mean_list):.5f}')
    elif args.task == 'rq2':
        for keys in ['init', 'inc']:
            cov_mean_list = []
            for result in results:
                cov_mean_list.append(result[keys])
            print('==============================================================')
            print(f'{keys} : {np.mean(cov_mean_list):.5f}')
    else:
        raise ValueError('Unknown task')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--scenario', '-s',
                        type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'imagenet', 'backdoor'],
                        help='scenario choice')
    parser.add_argument('--task', '-t',
                        type=str, default='rq1',
                        choices=['rq1', 'noise', 'rq2'],
                        help='task choice')
    parser.add_argument('--rq2_task',
                        type=str, default='poison',
                        choices=['randn', 'PGD', 'FGSM', 'poison'],
                        help='task choice for rq2')
    # parser.add_argument('--random_seed', '-rs',
    #                     type=int, default=0,
    #                     help='random seed')
    parser.add_argument('--repeat', '-r',
                        type=int, default=1,
                        help='repeat time')
    parser.add_argument('--mini',
                        action='store_true',
                        default=False,
                        help='mini')
    parser.add_argument('--batch_size', '-b',
                        type=int, default=200,
                        help='batch size')
    parser.add_argument('--setup_size', '-ss',
                        type=int, default=1000,
                        help='setup size')
    parser.add_argument('--trigger_info', '-ti',
                        type=str, default='/home/zjiae/Project/Causal-Coverage/data/badnet_trigger.pt',
                        help='trigger info')
    parser.add_argument('--backdoor_model',
                        type=str, default='/home/zjiae/Project/Causal-Coverage/output/model/badnet_vgg.pt',
                        help='backdoor model')
    args = parser.parse_args()
    print(args)

    # if args.random_seed >=0:
    #     random.seed(args.random_seed)
    #     np.random.seed(args.random_seed)
    #     torch.manual_seed(args.random_seed)
    #     torch.cuda.manual_seed(args.random_seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    main(args)
