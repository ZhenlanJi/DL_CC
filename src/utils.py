import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy
from scipy.stats import chisquare

from torchvision.models.resnet import Bottleneck
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    def __init__(self, data, targets, transform_list=None):
        data_X = data
        data_y = targets
        X_tensor, y_tensor = torch.tensor(data_X), torch.tensor(data_y)
        #X_tensor, y_tensor = Tensor(data_X), Tensor(data_y)
        # tensors = (X_tensor, y_tensor)
        assert all(X_tensor[0].size(0) == tensor.size(0)
                   for tensor in X_tensor)
        self.data = X_tensor
        self.targets = y_tensor
        self.classes = np.unique(targets)
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.data[index]

        if self.transforms:
            # for transform in self.transforms:
            #  x = transform(x)
            x = self.transforms(x)

        y = self.targets[index]

        return x, y

    def __len__(self):
        return self.data.size(0)


valid_instance_list = [
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.RNN, nn.LSTM, nn.GRU,
    nn.MaxPool2d, nn.MaxPool1d, nn.MaxPool3d,
]


def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            # or isinstance(module, Bottleneck)
            # or isinstance(module, nn.MaxPool2d)
            # or isinstance(module, nn.Conv3d)
            # or isinstance(module, nn.RNN)
            # or isinstance(module, nn.LSTM)
            # or isinstance(module, nn.GRU)
            )


def iterate_module(name, module, name_list, module_list):
    if is_valid(module):
        return name_list + [name], module_list + [module]
    else:
        # if len(list(module.named_children())):
        # specific for ResNet
        if len(list(module.named_children())) and name != 'downsample':
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module,
                                   name_list, module_list)
        return name_list, module_list


def get_model_layers(model):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [])
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' %
                       (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])

    return layer_dict


def get_layer_output_sizes(model, data, layer_name=None):
    output_sizes = {}
    hooks = []

    layer_dict = get_model_layers(model)

    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])
        # output_sizes[m_key] = list(output.size())

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        if type(data) is tuple:
            model(*data)
        else:
            model(data)
    finally:
        for h in hooks:
            h.remove()
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])

    return output_sizes


def get_layer_input_sizes(model, data, layer_name=None):
    input_sizes = {}
    hooks = []

    layer_dict = get_model_layers(model)

    def hook(module, input, output):
        module_idx = len(input_sizes)
        m_key = list(layer_dict)[module_idx]
        if type(input) is tuple:
            input = input[0]
        input_sizes[m_key] = list(input.size()[1:])
        # output_sizes[m_key] = list(output.size())

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        if type(data) is tuple:
            model(*data)
        else:
            model(data)
    finally:
        for h in hooks:
            h.remove()
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])

    return input_sizes


def get_layer_inout_sizes(model, data, layer_name=None):
    inout_sizes = {}
    hooks = []

    layer_dict = get_model_layers(model)

    def hook(module, input, output):
        module_idx = len(inout_sizes)
        m_key = list(layer_dict)[module_idx]
        if type(input) is tuple:
            input = input[0]
        inout_sizes[m_key] = (list(input.size()[1:]), list(output.size()[1:]))

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        if type(data) is tuple:
            model(*data)
        else:
            model(data)
    finally:
        for h in hooks:
            h.remove()
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])

    return inout_sizes


def get_layer_output(model, data, threshold=0.5, force_relu=False, is_origin=False):
    with torch.no_grad():
        layer_dict = get_model_layers(model)

        layer_output_dict = {}

        def hook(module, input, output):
            module_idx = len(layer_output_dict)
            m_key = list(layer_dict)[module_idx]
            if force_relu:
                output = F.relu(output)
            # (N, K, H, W) or (N, K)
            layer_output_dict[m_key] = output.detach()

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        # layer_output_list = []

        for layer, output in layer_output_dict.items():
            assert len(output.size()) == 2 or len(output.size()) == 4
            if not is_origin:
                if len(output.size()) == 4:  # (N, K, H, w)
                    layer_output_dict[layer] = output.mean((2, 3))
            # layer_output_dict[layer] = output.detach()
            # print(layer, ': ', output.size())
        return layer_output_dict


def get_layer_input(model, data, threshold=0.5, is_origin=False):
    with torch.no_grad():
        layer_dict = get_model_layers(model)

        layer_input_dict = {}

        def hook(module, input, output):
            module_idx = len(layer_input_dict)
            m_key = list(layer_dict)[module_idx]
            if type(input) is tuple:
                input = input[0]
            layer_input_dict[m_key] = input.detach()  # (N, K, H, W) or (N, K)

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        for layer, input in layer_input_dict.items():
            assert len(input.size()) == 2 or len(input.size()) == 4
            if not is_origin:
                if len(input.size()) == 4:  # (N, K, H, w)
                    layer_input_dict[layer] = input.mean((2, 3))
            # print(layer, ': ', output.size())
        return layer_input_dict


def get_layer_inout(model, data, threshold=0.5, is_origin=False):
    with torch.no_grad():
        layer_dict = get_model_layers(model)

        layer_inout_dict = {}

        def hook(module, input, output):
            module_idx = len(layer_inout_dict)
            m_key = list(layer_dict)[module_idx]
            if type(input) is tuple:
                input = input[0]
            layer_inout_dict[m_key] = (input.detach(), output.detach())

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        for layer, (input, output) in layer_inout_dict.items():
            assert len(input.size()) == 2 or len(input.size()) == 4
            assert len(output.size()) == 2 or len(output.size()) == 4
            if not is_origin:
                if len(input.size()) == 4:  # (N, K, H, w)
                    input = input.mean((2, 3))
                if len(output.size()) == 4:  # (N, K, H, w)
                    output = output.mean((2, 3))
                layer_inout_dict[layer] = (input, output)
                # print(layer, ': ', output.size())
        return layer_inout_dict


def get_selected_inout(model, data, selected, is_origin=False):
    with torch.no_grad():
        layer_dict = get_model_layers(model)

        layer_inout_dict = {}

        def hook(module, input, output):
            module_idx = len(layer_inout_dict)
            m_key = list(selected)[module_idx]
            if type(input) is tuple:
                input = input[0]
            layer_inout_dict[m_key] = (input.detach(), output.detach())

        hooks = []
        for layer_name in selected:
            module = layer_dict[layer_name]
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        for layer, (input, output) in layer_inout_dict.items():
            assert len(input.size()) == 2 or len(input.size()) == 4
            assert len(output.size()) == 2 or len(output.size()) == 4
            if not is_origin:
                if len(input.size()) == 4:  # (N, K, H, w)
                    input = input.mean((2, 3))
                if len(output.size()) == 4:  # (N, K, H, w)
                    output = output.mean((2, 3))
                layer_inout_dict[layer] = (input, output)
                # print(layer, ': ', output.size())
        return layer_inout_dict


def update_back_layer(model, data, front_layer, back_layer, new_output, is_origin=False):
    with torch.no_grad():
        layer_dict = get_model_layers(model)
        ret = []

        def update_front(module, input, output):
            return new_output

        def get_back(module, input, output):
            ret.append(output.detach())

        hooks = []
        hooks.append(
            layer_dict[front_layer].register_forward_hook(update_front))
        hooks.append(layer_dict[back_layer].register_forward_hook(get_back))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        assert len(ret) == 1
        assert len(ret[0].size()) == 2 or len(ret[0].size()) == 4
        if not is_origin:
            if len(ret[0].size()) == 4:
                return ret[0].mean((2, 3))
        return ret[0]


if __name__ == '__main__':
    pass
