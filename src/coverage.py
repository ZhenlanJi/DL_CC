import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pyflann import FLANN
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import chi2
import random
import numpy as np
from datetime import datetime
from causallearn.utils.cit import chisq


from estimator import Estimator, EstimatorFlatten
import utils
import models
import coverage
import train_model


def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    '''
    out_max = out.max()
    out_min = out.min()
    Note that the above implementation (from the offical release of DeepXplore)
    is incorrect when batch_size > 1
    '''
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


def Euclidean_distance(x, y):
    return ((x - y) ** 2).sum().sqrt()


'''
for SAs
var_threshold_dict = {
    'CIFAR10-resnet50': 1e-5,
    'CIFAR10-vgg16_bn': 1e-2,
    'CIFAR10-mobilenet_v2': 1e-5,
    'ImageNet-resnet50': 1e-5,
    'ImageNet-vgg16_bn': 1e-2,
    'ImageNet-mobilenet_v2': 1e-5,
}
'''


class LSA(object):
    def __init__(self, model, threshold, min_var, class_num, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.min_var = min_var
        self.class_num = class_num
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(
                layer_size[0]).cuda().type(torch.cuda.LongTensor)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).cuda()
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).cuda()

        self.data_count = 0
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Mean & Var on {num} batch...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            self.set_meam_var(data)
        self.set_mask()
        print(f'Building SA on {num} batch...')
        for i, (data, label) in enumerate(data_loader):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            label = label.cuda()
            self.build_SA(data, label)
        self.to_numpy()
        self.set_kde()
        # print('Building Coverage...')
        # for i, (data, label) in enumerate(tqdm(data_loader)):
        #     data = data.cuda()
        #     label = label.cuda()
        #     self.build_step(data, label)

    def build_step(self, data, label):
        cove_set = self.calculate(data, label)
        self.update(cove_set)

    def set_meam_var(self, data):
        if isinstance(data, tuple):
            batch_size = data[0].size(0)
        else:
            batch_size = data.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = (
                (self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
                + (self.data_count - batch_size) * ((layer_output -
                                                     self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (
                self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        # SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())
                              ] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [
                    SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        for k in self.SA_cache.keys():
            # self.kde_cache[k] = gaussian_kde(self.SA_cache[k].T)
            self.kde_cache[k] = KernelDensity(
                kernel='gaussian', bandwidth=0.2).fit(self.SA_cache[k])

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach(
        ).cpu().numpy()  # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            if (np.isnan(SA).any()) or (np.isinf(SA).any()):
                continue
            # lsa = np.asscalar(-self.kde_cache[int(label.cpu())
            #                                   ].logpdf(np.expand_dims(SA, 1)))
            lsa = np.asscalar(-self.kde_cache[int(label.cpu())
                                              ].score_samples(np.expand_dims(SA, 0)))
            if (not np.isnan(lsa)) and (not np.isinf(lsa)):
                cove_set.add(int(lsa / self.threshold))
            # cove_set.add(int(lsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(self.coverage_set)

    def coverage(self, is_covered):
        pass

    def all_coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new):
        new_rate = self.all_coverage(cove_set_new)
        return new_rate - self.current

    def save(self, path):
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.all_coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)


class DSA(object):
    def __init__(self, model, threshold, min_var, class_num, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.min_var = min_var
        self.class_num = class_num
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        self.SA_cache = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(
                layer_size[0]).cuda().type(torch.cuda.LongTensor)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).cuda()
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).cuda()

        self.data_count = 0
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Mean & Var on {num} batch...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            self.set_meam_var(data)
        self.set_mask()
        print(f'Building SA on {num} batch...')
        for i, (data, label) in enumerate(data_loader):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            label = label.cuda()
            self.build_SA(data, label)
        self.to_numpy()
        # print('Building Coverage...')
        # for i, (data, label) in enumerate(data_loader):
        #     data = data.cuda()
        #     label = label.cuda()
        #     self.build_step(data, label)

    def build_step(self, data, label):
        cove_set = self.calculate(data, label)
        self.update(cove_set)

    def set_meam_var(self, data):
        if isinstance(data, tuple):
            batch_size = data[0].size(0)
        else:
            batch_size = data.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = (
                (self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
                + (self.data_count - batch_size) * ((layer_output -
                                                     self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (
                self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())
                              ] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [
                    SA_batch[i].detach().cpu().numpy()]

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach(
        ).cpu().numpy()  # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).cuda() - torch.from_numpy(self.SA_cache[int(label.cpu())]).cuda(), dim=1)
            idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (
                self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.class_num):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA).cuda(
                        ) - torch.from_numpy(self.SA_cache[j]).cuda(),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
            # cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(self.coverage_set)

    def coverage(self, is_covered):
        pass

    def all_coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new):
        new_rate = self.all_coverage(cove_set_new)
        return new_rate - self.current

    def save(self, path):
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.all_coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)


class MDSA(object):
    def __init__(self, model, threshold, min_var, class_num, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.min_var = min_var
        self.class_num = class_num
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(
                layer_size[0]).cuda().type(torch.cuda.LongTensor)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).cuda()
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).cuda()
        self.data_count = 0
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Mean & Var on {num} batch...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            self.set_meam_var(data)
        self.set_mask()
        print('Building SA...')
        for i, (data, label) in enumerate(data_loader):
            if i >= num:
                break
            if isinstance(data, tuple):
                data = (data[0].cuda(), data[1].cuda())
            else:
                data = data.cuda()
            label = label.cuda()
            self.build_SA(data, label)
        # print('Building Coverage...')
        # for i, (data, label) in enumerate(tqdm(data_loader)):
        #     data = data.cuda()
        #     label = label.cuda()
        #     self.build_step(data, label)

    def build_step(self, data, label):
        cove_set = self.calculate(data, label)
        self.update(cove_set)

    def set_meam_var(self, data):
        if isinstance(data, tuple):
            batch_size = data[0].size(0)
        else:
            batch_size = data.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = (
                (self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
                + (self.data_count - batch_size) * ((layer_output -
                                                     self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (
                self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)
        self.estimator = Estimator(
            feature_num=feature_num, class_num=self.class_num, use_cuda=True)
        # class conditional mu and covariance

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        # SA_batch = SA_batch.cpu()
        # label_batch = label_batch.cpu()
        stat_dict = self.estimator.calculate(SA_batch, label_batch)
        self.estimator.update(stat_dict)

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = utils.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(
                layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        # SA_batch = SA_batch.cpu()
        mu = self.estimator.Ave[label_batch]
        covar = self.estimator.CoVariance[label_batch]

        covar_inv = torch.linalg.inv(covar)
        mdsa = (torch.bmm(torch.bmm((SA_batch - mu).unsqueeze(1),
                covar_inv), (SA_batch - mu).unsqueeze(2))).sqrt()
        # [bs, 1, n] x [bs, n, n] x [bs, n, 1]
        # [bs, 1]
        mdsa = mdsa.view(batch_size, -1)
        mdsa = mdsa[~torch.any(mdsa.isnan(), dim=1)]
        mdsa = mdsa[~torch.any(mdsa.isinf(), dim=1)]
        mdsa = mdsa.view(-1)
        if len(mdsa) > 0:
            mdsa_list = (mdsa / self.threshold).cpu().numpy().tolist()
            mdsa_list = [int(_mdsa) for _mdsa in mdsa_list]
            cove_set = set(mdsa_list)
            cove_set = self.coverage_set.union(cove_set)
        return cove_set

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(self.coverage_set)

    def coverage(self, is_covered):
        pass

    def all_coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new):
        new_rate = self.all_coverage(cove_set_new)
        return new_rate - self.current

    def save(self, path):
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'mu': self.estimator.Ave,
            'covar': self.estimator.CoVariance,
            'amount': self.estimator.Amount
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.estimator.Ave = state['mu']
        self.estimator.CoVariance = state['covar']
        self.estimator.Amount = state['amount']
        loaded_cov = self.all_coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)


class NCS(object):
    '''
    Neuron Coverage
    1. rescale neuron outputs to [0, 1]
    2. a neuron is activated if its output is greater than the threshold
    '''

    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(
                layer_size[0]).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class NC(object):
    '''
    Neuron Coverage
    1. a neuron is activated if its output is greater than the threshold
    '''

    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(
                layer_size[0]).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            mask_index = layer_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class KMNC(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_multisec_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_multisec_dict[layer_name] = torch.zeros(
                (num_neuron, k + 1)).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'multisec_cove_dict': self.coverage_multisec_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Range on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_range(data)

        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(data_loader):
            if i < num:
                data = data.cuda()
                self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        multisec_cove_dict = {}
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (
                layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(
                num_neuron, self.k + 1).cuda().type(torch.cuda.BoolTensor)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            multisec_output = torch.ceil(
                (layer_output - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index
            # (1, k), index 0 indicates out-of-range output

            index = tuple(
                [torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cove_dict[layer_name] = multisec_covered | self.coverage_multisec_dict[layer_name]

        return {
            'multisec_cove_dict': multisec_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
            for l in self.coverage_multisec_dict.keys():
                self.coverage_multisec_dict[l] |= all_cove_dict[k][l]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        multisec_covered = all_covered_dict['multisec_covered']

        num_neuron = multisec_covered.size(0)
        multisec_num_covered = torch.sum(multisec_covered[:, 1:])
        multisec_rate = multisec_num_covered / (num_neuron * self.k)

        return multisec_rate.item()

    def all_coverage(self, all_cove_dict):
        multisec_cove_dict = all_cove_dict['multisec_cove_dict']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total
        return multisec_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_multisec_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class SNAC(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_upper_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_upper_dict[layer_name] = torch.zeros(
                num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'upper_cove_dict': self.coverage_upper_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Range on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_range(data)

        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(data_loader):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        upper_cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cove_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'upper_cove_dict': upper_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
            for l in self.coverage_upper_dict.keys():
                self.coverage_upper_dict[l] |= all_cove_dict[k][l]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        upper_covered = all_covered_dict['upper_covered']
        upper_rate = upper_covered.sum() / len(upper_covered)
        return upper_rate.item()

    def all_coverage(self, all_cove_dict):
        upper_cove_dict = all_cove_dict['upper_cove_dict']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_upper_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class NBC(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_lower_dict = {}
        self.coverage_upper_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_lower_dict[layer_name] = torch.zeros(
                num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.coverage_upper_dict[layer_name] = torch.zeros(
                num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'lower_cove_dict': self.coverage_lower_dict,
            'upper_cove_dict': self.coverage_upper_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Range on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_range(data)

        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(data_loader):
            if i < num:
                data = data.cuda()
                self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]

            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_lower_dict[layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'lower_cove_dict': lower_cove_dict,
            'upper_cove_dict': upper_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
        for l in self.coverage_lower_dict.keys():
            self.coverage_lower_dict[l] |= all_cove_dict['lower_cove_dict'][l]
        for l in self.coverage_upper_dict.keys():
            self.coverage_upper_dict[l] |= all_cove_dict['upper_cove_dict'][l]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        lower_covered = all_covered_dict['lower_covered']
        upper_covered = all_covered_dict['upper_covered']

        lower_rate = lower_covered.sum() / len(lower_covered)
        upper_rate = upper_covered.sum() / len(upper_covered)

        return (lower_rate + upper_rate).item() / 2

    def all_coverage(self, all_cove_dict):
        lower_cove_dict = all_cove_dict['lower_cove_dict']
        upper_cove_dict = all_cove_dict['upper_cove_dict']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cove_dict.keys():
            lower_covered = lower_cove_dict[layer_name]
            upper_covered = upper_cove_dict[layer_name]

            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()

            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = lower_cove / lower_total
        upper_rate = upper_cove / upper_total
        return (lower_rate + upper_rate).item() / 2

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     state = {
    #         'lower': self.coverage_lower_dict,
    #         'upper': self.coverage_upper_dict
    #     }
    #     torch.save(state, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class TKNC(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(
                num_neuron).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(
                min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).cuda()
            index = tuple(
                [torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1

            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class TKNP(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.layer_pattern = {}
        self.network_pattern = set()
        self.current = 0
        for (layer_name, layer_size) in layer_size_dict.items():
            self.layer_pattern[layer_name] = set()

    def build(self, data_loader, num=10):
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        layer_pat = {}
        layer_output_dict = utils.get_layer_output(self.model, data)
        topk_idx_list = []
        for (layer_name, layer_output) in layer_output_dict.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(
                min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            # idx: (batch_size, k)
            pat = set([str(s) for s in list(idx[:, ])])
            topk_idx_list.append(idx)
            layer_pat[layer_name] = set.union(
                pat, self.layer_pattern[layer_name])
        network_topk_idx = torch.cat(topk_idx_list, 1)
        network_pat = set([str(s) for s in list(network_topk_idx[:, ])])
        network_pat = set.union(network_pat, self.network_pattern)
        return {
            'layer_pat': layer_pat,
            'network_pat': network_pat
        }

    def update(self, all_pat_dict, delta=None):
        layer_pat = all_pat_dict['layer_pat']
        network_pat = all_pat_dict['network_pat']
        self.layer_pattern = layer_pat
        self.network_pattern = network_pat
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_pat_dict)

    def coverage(self, pattern):
        return len(pattern)

    def all_coverage(self, all_pat_dict):
        network_pat = all_pat_dict['network_pat']
        return len(network_pat)

    def gain(self, all_pat_dict):
        new_rate = self.all_coverage(all_pat_dict)
        return new_rate - self.current

    def save(self, path):
        pass

    def load(self, path):
        pass


class CC(object):
    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.distant_dict = {}
        self.flann_dict = {}

        for (layer_name, layer_size) in layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        dis_dict = self.calculate(data)
        self.update(dis_dict)

    def update(self, dis_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dis_dict[layer_name]
            self.flann_dict[layer_name].build_index(
                np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.all_coverage(dis_dict)

    def calculate(self, data):
        layer_output_dict = utils.get_layer_output(self.model, data)
        dis_dict = {}
        for (layer_name, layer_output) in layer_output_dict.items():
            dis_dict[layer_name] = []
            for single_output in layer_output:
                single_output = single_output.cpu().numpy()
                if len(self.distant_dict[layer_name]) > 0:
                    _, approx_distances = self.flann_dict[layer_name].nn_index(
                        np.expand_dims(single_output, 0), num_neighbors=1)
                    exact_distances = [
                        np.sum(np.square(single_output - distant_vec))
                        for distant_vec in self.distant_dict[layer_name]
                    ]
                    buffer_distances = [
                        np.sum(np.square(single_output - buffer_vec))
                        for buffer_vec in dis_dict[layer_name]
                    ]
                    nearest_distance = min(
                        exact_distances + approx_distances.tolist() + buffer_distances)
                    if nearest_distance > self.threshold:
                        dis_dict[layer_name].append(single_output)
                else:
                    self.flann_dict[layer_name].build_index(single_output)
                    self.distant_dict[layer_name].append(single_output)
        return dis_dict

    def coverage(self, distant):
        return len(distant)

    def all_coverage(self, dis_dict):
        total = 0
        for layer_name in dis_dict.keys():
            total += len(dis_dict[layer_name])
        return total

    def gain(self, dis_dict):
        increased = self.all_coverage(dis_dict)
        return increased

    def save(self, path):
        pass

    def load(self, path):
        pass


class CGRC(object):
    # Causal Graph Reconstruction Coverage
    def __init__(self, model, k, layer_size_dict, p_value=0.05, chi2_test_threshold=5, is_naive=False, is_plus=False):
        assert len(layer_size_dict) > 0
        self.p_value = p_value
        self.chi2_test_threshold = chi2_test_threshold
        self.model = model
        self.count = 0
        self.k = k
        self.dof = (k-1)*(k-1)
        self.critical = chi2.ppf(1-self.p_value, self.dof)
        self.is_naive = is_naive
        self.is_plus = is_plus
        self.range_dict = {}
        self.input_mean_dict = {}
        self.cov_mean_dict = {}
        self.covariance_dict = {}
        self.coverage_edge_dict = {}
        self.count_edge_dict = {}
        self.bound_dict = {}
        # self.origin_sec_dict = {}
        # self.new_sec.dict = {}

        # # add the input to first layer edge
        # first_layer_name = list(layer_size_dict.keys())[0]
        # first_num_neuron = layer_size_dict[first_layer_name][1][0]
        # self.coverage_edge_dict["Input_"+first_layer_name] = torch.zeros(
        #     (1, first_num_neuron)).cuda().type(torch.cuda.BoolTensor)

        for i, (layer_name, layer_size) in enumerate(layer_size_dict.items()):
            input_size, output_size = layer_size
            num_neuron = output_size[0]
            if i > 0:
                edge = list(layer_size_dict.keys())[i-1] + '__' + layer_name
                front_layer_num_neuron = input_size[0]
                self.coverage_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, num_neuron)).cuda().type(torch.cuda.BoolTensor)
                self.count_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, num_neuron, k, k)).cuda().type(torch.cuda.LongTensor)
                self.bound_dict[edge] = torch.zeros(
                    (front_layer_num_neuron)).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
            self.input_mean_dict[layer_name] = torch.zeros(
                input_size[0]).cuda().type(torch.cuda.FloatTensor)
            self.cov_mean_dict[layer_name] = torch.zeros(
                (input_size[0], input_size[0])).cuda().type(torch.cuda.FloatTensor)
            self.covariance_dict[layer_name] = torch.zeros(
                (input_size[0], input_size[0])).cuda().type(torch.cuda.FloatTensor)
            # self.origin_sec_dict[layer_name] = torch.zeros(
            #     (num_neuron, k+1)).cuda().type(torch.cuda.IntTensor)
            # self.new_sec_dict[layer_name] = torch.zeros(
            #     (input_size[0], num_neuron, k+1)).cuda().type(torch.cuda.IntTensor)

        self.coverage_dict = {
            'edge_cove_dict': self.coverage_edge_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Contex on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_context(data)
        self.calc_covariance()
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_context(self, data):
        # input and output have reduced to batch_size*neuron_num(or input_num)
        layer_inout_dict = utils.get_layer_inout(self.model, data)
        for layer_name, (layer_input, layer_output) in layer_inout_dict.items():
            # set range
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

            # set mean
            self.input_mean_dict[layer_name] = (self.input_mean_dict[layer_name] * self.count +
                                                layer_input.sum(0)) / (self.count + data.shape[0])
            # input_num * input_num
            cov_sum = torch.matmul(layer_input.T, layer_input)
            self.cov_mean_dict[layer_name] = (
                self.cov_mean_dict[layer_name]*self.count + cov_sum)/(self.count + data.shape[0])
            #input_num * input_num
        self.count += data.shape[0]

    def calc_covariance(self):
        for layer_name in self.cov_mean_dict.keys():
            mean_dict = self.input_mean_dict[layer_name].view(
                1, -1)  # 1 * input_num
            self.covariance_dict[layer_name] = (self.cov_mean_dict[layer_name] -
                                                torch.matmul(mean_dict.T, mean_dict)).fill_diagonal_(0)
            #input_num * input_num

    def count_sec_cov(self, layer_name, outputs):
        [l_bound, u_bound] = self.range_dict[layer_name]
        num_neuron = outputs.size(1)
        multisec_index = (u_bound > l_bound) & (
            outputs >= l_bound) & (outputs <= u_bound)
        div_index = u_bound > l_bound
        div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
        multisec_output = torch.ceil(
            (outputs - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index  # batch_size * num_neuron
        # (1, k), index 0 indicates out-of-range output. to work as index, must be LongTensor

        # multisec_covered = F.one_hot(multisec_output, self.k + 1).sum(0)
        return multisec_output

    def calculate(self, data):
        edge_coverage_dict = {}
        layer_inout_dict = utils.get_layer_inout(
            self.model, data, is_origin=True)
        layer_dict = utils.get_model_layers(self.model)
        for i, (layer_name, (layer_input, layer_output)) in enumerate(layer_inout_dict.items()):
            if i == 0:
                continue
            layer = layer_dict[layer_name]
            front_layer_name = list(layer_dict.keys())[i-1]
            front_layer_output = layer_inout_dict[front_layer_name][1]
            if len(front_layer_output.size()) == 4:
                front_layer_output = front_layer_output.mean((2, 3))
            front_layer_sec = self.count_sec_cov(
                front_layer_name, front_layer_output)  # batch_size * neuron_num

            # shape alignment
            ratio = layer_input.shape[1]/front_layer_sec.shape[1]
            if front_layer_sec.shape[1] < layer_input.shape[1]:
                front_layer_sec = torch.repeat_interleave(
                    front_layer_sec, int(ratio), dim=1)
            elif front_layer_sec.shape[1] > layer_input.shape[1]:
                front_layer_sec = front_layer_sec.view(
                    front_layer_sec.shape[0], -1, int(1/ratio)).mean(2)
            assert front_layer_sec.shape[1] == layer_input.shape[1]

            edge = front_layer_name+'__'+layer_name

            if self.is_naive:
                edge_coverage_dict[edge] = torch.zeros(
                    (front_layer_sec.shape[1], layer_output.shape[1])).cuda().type(torch.cuda.BoolTensor)
                if len(layer_output.size()) == 4:
                    layer_output = layer_output.mean((2, 3))
                layer_sec = self.count_sec_cov(
                    layer_name, layer_output)  # batch_size * neuron_num
                for n in range(front_layer_sec.shape[1]):
                    for m in range(layer_sec.shape[1]):
                        test_data = torch.cat(
                            (front_layer_sec, layer_sec[:, m].view(-1, 1)), dim=1).cpu().detach().numpy().astype(np.int32)
                        conditioning_set_list = list(
                            range(n))+list(range(n+1, front_layer_sec.shape[1]))
                        selected = random.sample(conditioning_set_list, min(
                            len(conditioning_set_list), 10))
                        p_value = chisq(
                            test_data, X=n, Y=front_layer_sec.shape[1], conditioning_set=selected)
                        edge_coverage_dict[edge][n, m] = bool(
                            p_value <= self.p_value) | self.coverage_edge_dict[edge][n, m]

            else:
                coeff = torch.abs(self.covariance_dict[layer_name]) / \
                    (torch.abs(self.covariance_dict[layer_name]).sum(
                        1).view(-1, 1))  # input_num*input_num
                coeff = 1-coeff
                for n in range(layer_input.shape[1]):
                    if len(layer_input.size()) == 2:
                        # batch_size * num_neuron
                        new_input = layer_input*coeff[n]
                    elif len(layer_input.size()) == 4:
                        new_input = coeff[n].repeat(layer_input.shape[0], 1)[
                            :, :, None, None]*layer_input  # batch_size * num_neuron * x * y
                    new_output = layer(new_input)
                    if len(new_output.size()) == 4:
                        new_output = new_output.mean((2, 3))
                    new_layer_sec = self.count_sec_cov(
                        layer_name, new_output)  # batch_size * num_neuron

                    # count edge sec distribution
                    edge_sec = (front_layer_sec[:, n] *
                                (self.k+1)).view(-1, 1)+new_layer_sec
                    edge_sec_count = F.one_hot(
                        edge_sec, (self.k + 1)*(self.k+1)
                    ).sum(0).view(-1, self.k+1, self.k+1)  # neuron_num * (k+1) * (k+1)
                    # filter out out of range case
                    self.count_edge_dict[edge][n] += edge_sec_count[:, 1:, 1:]
                    if self.is_plus:
                        self.bound_dict[edge][n] |= edge_sec_count[:, 0, 0].sum(
                            0) > 0
                    # self.bound_dict[edge][n] |= edge_sec_count[:, 0, 0].type(torch.cuda.BoolTensor)

                # count edge coverage
                # (input_num, neuron_num, k, k)
                observed = self.count_edge_dict[edge]
                row_sums = observed.type(torch.cuda.FloatTensor).mean(
                    3, keepdim=True)  # (input_num, neuron_num, k, 1)
                col_sums = observed.type(torch.cuda.FloatTensor).mean(
                    2, keepdim=True)  # (input_num, neuron_num, 1, k)
                # (input_num, neuron_num, k, k)
                expected = row_sums.mul(col_sums)
                mask_1 = ((expected < self.chi2_test_threshold).sum((2, 3))) <= 0
                chi2_values = ((observed - expected)**2 / expected).sum((2, 3))
                mask_2 = (chi2_values >= self.critical)

                if self.is_plus:
                    edge_coverage_dict[edge] = (mask_1 & mask_2 &
                                                self.bound_dict[edge].view(-1, 1).repeat(1, mask_1.shape[1])) | \
                        (self.coverage_edge_dict[edge])
                    # edge_coverage_dict[edge] = (mask_1 & mask_2 & self.bound_dict[edge]) | (
                    #     self.coverage_edge_dict[edge])
                else:
                    edge_coverage_dict[edge] = (mask_1 & mask_2) | (
                        self.coverage_edge_dict[edge])

        return {
            'edge_cove_dict': edge_coverage_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
            for edge in self.coverage_edge_dict.keys():
                self.coverage_edge_dict[edge] |= all_cove_dict[k][edge]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    # def coverage(self, all_covered_dict):
    #     multisec_covered = all_covered_dict['edge_cove_dict']

    #     num_neuron = multisec_covered.size(0)
    #     multisec_num_covered = torch.sum(multisec_covered[:, 1:])
    #     multisec_rate = multisec_num_covered / (num_neuron * self.k)

    #     return multisec_rate.item()

    def all_coverage(self, all_cove_dict):
        coverage_dict = all_cove_dict['edge_cove_dict']
        (edge_cove, edge_total) = (0, 0)
        for layer_name in coverage_dict.keys():
            layer_edge_covered = coverage_dict[layer_name]
            num_edge = layer_edge_covered.shape[0]*layer_edge_covered.shape[1]
            edge_cove += layer_edge_covered.sum()
            edge_total += num_edge
        covered_rate = edge_cove / edge_total
        return covered_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_multisec_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'mean': self.input_mean_dict,
            'cov_mean': self.cov_mean_dict,
            'covariance': self.covariance_dict,
            'edge_cov': self.coverage_edge_dict,
            'edge_count': self.count_edge_dict,
            'bound': self.bound_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']
        self.input_mean_dict = state['mean']
        self.cov_mean_dict = state['cov_mean']
        self.covariance_dict = state['covariance']
        self.coverage_edge_dict = state['edge_cov']
        self.count_edge_dict = state['edge_count']
        self.bound_dict = state['bound']

        # loaded_cov = self.all_coverage(self.coverage_dict)
        self.update(self.coverage_dict)
        print(f'Loaded coverage: {self.current}')


class CGRC_mini(object):
    # Causal Graph Reconstruction Coverage select only 2~n layers, default value is 3 layers
    # This method is implemented by extracting sub_model from the whole model, it works well for
    # sequential model, however, it is not good for complex model. We recommend to implement
    # this method based on register_forward_hook for complex model. See CGRC_mini_v2.
    def __init__(self, model, k, layer_size_dict, sub_models, selected_layers, p_value=0.05, chi2_test_threshold=5):
        assert len(layer_size_dict) > 0
        self.p_value = p_value
        self.chi2_test_threshold = chi2_test_threshold
        self.model = model
        self.sub_models = sub_models
        self.selected_layers = selected_layers
        self.count = 0
        self.k = k
        self.dof = (k-1)*(k-1)
        self.critical = chi2.ppf(1-self.p_value, self.dof)
        self.range_dict = {}
        self.activate_mean_dict = {}
        self.cov_mean_dict = {}
        self.covariance_dict = {}
        self.coverage_edge_dict = {}
        self.count_edge_dict = {}

        for i, layer_name in enumerate(selected_layers):
            layer_size = layer_size_dict[layer_name]
            _, output_size = layer_size
            num_neuron = output_size[0]
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
            if i < len(selected_layers)-1:
                edge = layer_name + '__' + selected_layers[i+1]
                front_layer_num_neuron = num_neuron
                back_layer_num_neuron = layer_size_dict[selected_layers[i+1]][1][0]
                self.coverage_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, back_layer_num_neuron)).cuda().type(torch.cuda.BoolTensor)
                self.count_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, back_layer_num_neuron, k, k)).cuda().type(torch.cuda.LongTensor)

                self.activate_mean_dict[layer_name] = torch.zeros(
                    front_layer_num_neuron).cuda().type(torch.cuda.FloatTensor)
                self.cov_mean_dict[layer_name] = torch.zeros(
                    (front_layer_num_neuron, front_layer_num_neuron)).cuda().type(torch.cuda.FloatTensor)
                self.covariance_dict[layer_name] = torch.zeros(
                    (front_layer_num_neuron, front_layer_num_neuron)).cuda().type(torch.cuda.FloatTensor)

        self.coverage_dict = {
            'edge_cove_dict': self.coverage_edge_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Contex on {num} batch...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_context(data)
        self.calc_covariance()
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_context(self, data):
        # input and output have reduced to batch_size*neuron_num(or input_num)
        layer_inout_dict = utils.get_selected_inout(
            self.model, data, self.selected_layers)
        for i, layer_name in enumerate(self.selected_layers):
            _, layer_output = layer_inout_dict[layer_name]
            # set range
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

            if i < len(self.selected_layers)-1:
                # set mean
                self.activate_mean_dict[layer_name] = (self.activate_mean_dict[layer_name] * self.count +
                                                       layer_output.sum(0)) / (self.count + data.shape[0])
                # num_neuron * num_neuron
                cov_sum = torch.matmul(layer_output.T, layer_output)
                self.cov_mean_dict[layer_name] = (
                    self.cov_mean_dict[layer_name]*self.count + cov_sum)/(self.count + data.shape[0])
                #input_num * input_num
        self.count += data.shape[0]

    def calc_covariance(self):
        for layer_name in self.cov_mean_dict.keys():
            mean_dict = self.activate_mean_dict[layer_name].view(
                1, -1)  # 1 * input_num
            self.covariance_dict[layer_name] = (self.cov_mean_dict[layer_name] -
                                                torch.matmul(mean_dict.T, mean_dict)).fill_diagonal_(0)
            #input_num * input_num

    def count_sec_cov(self, layer_name, outputs):
        [l_bound, u_bound] = self.range_dict[layer_name]
        num_neuron = outputs.size(1)
        multisec_index = (u_bound > l_bound) & (
            outputs >= l_bound) & (outputs <= u_bound)
        div_index = u_bound > l_bound
        div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
        multisec_output = torch.ceil(
            (outputs - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index  # batch_size * num_neuron
        # (1, k), index 0 indicates out-of-range output. to work as index, must be LongTensor

        # multisec_covered = F.one_hot(multisec_output, self.k + 1).sum(0)
        return multisec_output

    def calculate(self, data):
        edge_coverage_dict = {}
        layer_inout_dict = utils.get_selected_inout(
            self.model, data, self.selected_layers, is_origin=True)
        for i, layer_name in enumerate(self.selected_layers[:-1]):
            sub_model = self.sub_models[i]
            sub_model.eval()

            front_layer_name = layer_name
            back_layer_name = self.selected_layers[i+1]

            front_layer_output = layer_inout_dict[front_layer_name][1]
            if len(front_layer_output.size()) == 4:
                front_layer_reduced = front_layer_output.mean((2, 3))
            else:
                front_layer_reduced = front_layer_output
            front_layer_sec = self.count_sec_cov(
                front_layer_name, front_layer_reduced)  # batch_size * neuron_num

            edge = front_layer_name+'__'+back_layer_name

            coeff = torch.abs(self.covariance_dict[layer_name]) / \
                (torch.abs(self.covariance_dict[layer_name]).sum(
                    1).view(-1, 1))  # input_num*input_num
            coeff = 1-coeff
            for n in range(front_layer_output.shape[1]):
                if len(front_layer_output.size()) == 2:
                    # batch_size * num_neuron
                    new_input = front_layer_output*coeff[n]
                elif len(front_layer_output.size()) == 4:
                    new_input = coeff[n].repeat(front_layer_output.shape[0], 1)[
                        :, :, None, None]*front_layer_output  # batch_size * num_neuron * x * y
                new_output = sub_model(new_input)
                if len(new_output.size()) == 4:
                    new_output = new_output.mean((2, 3))
                new_layer_sec = self.count_sec_cov(
                    back_layer_name, new_output)  # batch_size * num_neuron

                # count edge sec distribution
                edge_sec = (front_layer_sec[:, n] *
                            (self.k+1)).view(-1, 1)+new_layer_sec
                edge_sec_count = F.one_hot(
                    edge_sec, (self.k + 1)*(self.k+1)
                ).sum(0).view(-1, self.k+1, self.k+1)  # neuron_num * (k+1) * (k+1)
                # filter out out of range case
                self.count_edge_dict[edge][n] += edge_sec_count[:, 1:, 1:]

            # count edge coverage
            # (input_num, neuron_num, k, k)
            observed = self.count_edge_dict[edge]
            row_sums = observed.type(torch.cuda.FloatTensor).mean(
                3, keepdim=True)  # (input_num, neuron_num, k, 1)
            col_sums = observed.type(torch.cuda.FloatTensor).mean(
                2, keepdim=True)  # (input_num, neuron_num, 1, k)
            # (input_num, neuron_num, k, k)
            expected = row_sums.mul(col_sums)
            mask_1 = ((expected < self.chi2_test_threshold).sum((2, 3))) <= 0
            chi2_values = ((observed - expected)**2 / expected).sum((2, 3))
            mask_2 = (chi2_values >= self.critical)
            edge_coverage_dict[edge] = (mask_1 & mask_2) | (
                self.coverage_edge_dict[edge])

        return {
            'edge_cove_dict': edge_coverage_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
            for edge in self.coverage_edge_dict.keys():
                self.coverage_edge_dict[edge] |= all_cove_dict[k][edge]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def all_coverage(self, all_cove_dict):
        coverage_dict = all_cove_dict['edge_cove_dict']
        (edge_cove, edge_total) = (0, 0)
        for layer_name in coverage_dict.keys():
            layer_edge_covered = coverage_dict[layer_name]
            num_edge = layer_edge_covered.shape[0]*layer_edge_covered.shape[1]
            edge_cove += layer_edge_covered.sum()
            edge_total += num_edge
        covered_rate = edge_cove / edge_total
        return covered_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_multisec_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'mean': self.activate_mean_dict,
            'cov_mean': self.cov_mean_dict,
            'covariance': self.covariance_dict,
            'edge_cov': self.coverage_edge_dict,
            'edge_count': self.count_edge_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.activate_mean_dict = state['mean']
        self.cov_mean_dict = state['cov_mean']
        self.covariance_dict = state['covariance']
        self.coverage_edge_dict = state['edge_cov']
        self.count_edge_dict = state['edge_count']
        self.coverage_dict = state['coverage']

        # loaded_cov = self.all_coverage(self.coverage_dict)
        self.update(self.coverage_dict)
        print(f'Loaded coverage: {self.current}')


class CGRC_mini_v2(object):
    # Causal Graph Reconstruction Coverage select only 2~n layers, default value is 3 layers
    # This method is implemented based on register_forward_hook. It works well for even complex
    # models, but may result in tremendous computational cost when n larger than 2.
    def __init__(self, model, k, layer_size_dict, selected_layers, p_value=0.05, chi2_test_threshold=5):
        assert len(layer_size_dict) > 0
        self.p_value = p_value
        self.chi2_test_threshold = chi2_test_threshold
        self.model = model
        self.selected_layers = selected_layers
        self.count = 0
        self.k = k
        self.dof = (k-1)*(k-1)
        self.critical = chi2.ppf(1-self.p_value, self.dof)
        self.range_dict = {}
        self.activate_mean_dict = {}
        self.cov_mean_dict = {}
        self.covariance_dict = {}
        self.coverage_edge_dict = {}
        self.count_edge_dict = {}

        for i, layer_name in enumerate(selected_layers):
            layer_size = layer_size_dict[layer_name]
            _, output_size = layer_size
            num_neuron = output_size[0]
            self.range_dict[layer_name] = [torch.ones(
                num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
            if i < len(selected_layers)-1:
                edge = layer_name + '__' + selected_layers[i+1]
                front_layer_num_neuron = num_neuron
                back_layer_num_neuron = layer_size_dict[selected_layers[i+1]][1][0]
                self.coverage_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, back_layer_num_neuron)).cuda().type(torch.cuda.BoolTensor)
                self.count_edge_dict[edge] = torch.zeros(
                    (front_layer_num_neuron, back_layer_num_neuron, k, k)).cuda().type(torch.cuda.LongTensor)

                self.activate_mean_dict[layer_name] = torch.zeros(
                    front_layer_num_neuron).cuda().type(torch.cuda.FloatTensor)
                self.cov_mean_dict[layer_name] = torch.zeros(
                    (front_layer_num_neuron, front_layer_num_neuron)).cuda().type(torch.cuda.FloatTensor)
                self.covariance_dict[layer_name] = torch.zeros(
                    (front_layer_num_neuron, front_layer_num_neuron)).cuda().type(torch.cuda.FloatTensor)

        self.coverage_dict = {
            'edge_cove_dict': self.coverage_edge_dict
        }
        self.current = 0

    def build(self, data_loader, num=10):
        print(f'Building Contex on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.set_context(data)
        self.calc_covariance()
        print(f'Building Coverage on {num} batch...')
        for i, (data, _) in enumerate(tqdm(data_loader)):
            if i >= num:
                break
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_context(self, data):
        # input and output have reduced to batch_size*neuron_num(or input_num)
        layer_inout_dict = utils.get_selected_inout(
            self.model, data, self.selected_layers)
        for i, layer_name in enumerate(self.selected_layers):
            _, layer_output = layer_inout_dict[layer_name]
            # set range
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * \
                cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * \
                cur_max + ~is_greater * self.range_dict[layer_name][1]

            if i < len(self.selected_layers)-1:
                # set mean
                self.activate_mean_dict[layer_name] = (self.activate_mean_dict[layer_name] * self.count +
                                                       layer_output.sum(0)) / (self.count + data.shape[0])
                # num_neuron * num_neuron
                cov_sum = torch.matmul(layer_output.T, layer_output)
                self.cov_mean_dict[layer_name] = (
                    self.cov_mean_dict[layer_name]*self.count + cov_sum)/(self.count + data.shape[0])
                #input_num * input_num
        self.count += data.shape[0]

    def calc_covariance(self):
        for layer_name in self.cov_mean_dict.keys():
            mean_dict = self.activate_mean_dict[layer_name].view(
                1, -1)  # 1 * input_num
            self.covariance_dict[layer_name] = (self.cov_mean_dict[layer_name] -
                                                torch.matmul(mean_dict.T, mean_dict)).fill_diagonal_(0)
            #input_num * input_num

    def count_sec_cov(self, layer_name, outputs):
        [l_bound, u_bound] = self.range_dict[layer_name]
        num_neuron = outputs.size(1)
        multisec_index = (u_bound > l_bound) & (
            outputs >= l_bound) & (outputs <= u_bound)
        div_index = u_bound > l_bound
        div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
        multisec_output = torch.ceil(
            (outputs - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index  # batch_size * num_neuron
        # (1, k), index 0 indicates out-of-range output. to work as index, must be LongTensor

        # multisec_covered = F.one_hot(multisec_output, self.k + 1).sum(0)
        return multisec_output

    def calculate(self, data):
        edge_coverage_dict = {}
        layer_inout_dict = utils.get_selected_inout(
            self.model, data, self.selected_layers, is_origin=True)
        for i, layer_name in enumerate(self.selected_layers[:-1]):
            front_layer_name = layer_name
            back_layer_name = self.selected_layers[i+1]

            front_layer_output = layer_inout_dict[front_layer_name][1]
            if len(front_layer_output.size()) == 4:
                front_layer_reduced = front_layer_output.mean((2, 3))
            else:
                front_layer_reduced = front_layer_output
            front_layer_sec = self.count_sec_cov(
                front_layer_name, front_layer_reduced)  # batch_size * neuron_num

            edge = front_layer_name+'__'+back_layer_name

            coeff = torch.abs(self.covariance_dict[layer_name]) / \
                (torch.abs(self.covariance_dict[layer_name]).sum(
                    1).view(-1, 1))  # input_num*input_num
            coeff = 1-coeff
            for n in range(front_layer_output.shape[1]):
                if len(front_layer_output.size()) == 2:
                    # batch_size * num_neuron
                    new_input = front_layer_output*coeff[n]
                elif len(front_layer_output.size()) == 4:
                    new_input = coeff[n].repeat(front_layer_output.shape[0], 1)[
                        :, :, None, None]*front_layer_output  # batch_size * num_neuron * x * y
                new_output = utils.update_back_layer(
                    self.model, data, front_layer_name, back_layer_name, new_input.detach(), is_origin=True)
                if len(new_output.size()) == 4:
                    new_output = new_output.mean((2, 3))
                new_layer_sec = self.count_sec_cov(
                    back_layer_name, new_output)  # batch_size * num_neuron

                # count edge sec distribution
                edge_sec = (front_layer_sec[:, n] *
                            (self.k+1)).view(-1, 1)+new_layer_sec
                edge_sec_count = F.one_hot(
                    edge_sec, (self.k + 1)*(self.k+1)
                ).sum(0).view(-1, self.k+1, self.k+1)  # neuron_num * (k+1) * (k+1)
                # filter out out of range case
                self.count_edge_dict[edge][n] += edge_sec_count[:, 1:, 1:]

            # count edge coverage
            # (input_num, neuron_num, k, k)
            observed = self.count_edge_dict[edge]
            row_sums = observed.type(torch.cuda.FloatTensor).mean(
                3, keepdim=True)  # (input_num, neuron_num, k, 1)
            col_sums = observed.type(torch.cuda.FloatTensor).mean(
                2, keepdim=True)  # (input_num, neuron_num, 1, k)
            # (input_num, neuron_num, k, k)
            expected = row_sums.mul(col_sums)
            mask_1 = ((expected < self.chi2_test_threshold).sum((2, 3))) <= 0
            chi2_values = ((observed - expected)**2 / expected).sum((2, 3))
            mask_2 = (chi2_values >= self.critical)
            edge_coverage_dict[edge] = (mask_1 & mask_2) | (
                self.coverage_edge_dict[edge])

        return {
            'edge_cove_dict': edge_coverage_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
            for edge in self.coverage_edge_dict.keys():
                self.coverage_edge_dict[edge] |= all_cove_dict[k][edge]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def all_coverage(self, all_cove_dict):
        coverage_dict = all_cove_dict['edge_cove_dict']
        (edge_cove, edge_total) = (0, 0)
        for layer_name in coverage_dict.keys():
            layer_edge_covered = coverage_dict[layer_name]
            num_edge = layer_edge_covered.shape[0]*layer_edge_covered.shape[1]
            edge_cove += layer_edge_covered.sum()
            edge_total += num_edge
        covered_rate = edge_cove / edge_total
        return covered_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_multisec_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


if __name__ == '__main__':
    img_channel = 1
    img_size = 32
    img_size = 32
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    test_save = {}

    state_dict = torch.load(
        "/home/zjiae/Project/Causal-Coverage/output/model/mnist_lenet_5.pt")
    model = models.LeNet_5()
    model.load_state_dict(state_dict)
    train_loader, test_loader, val_loader = train_model.get_dataloaders(
        'mnist', None, 100, 0)
    data_list = []
    for image, label in test_loader:
        data_list.append(image.to(DEVICE))
        # data_list.append((image.to(DEVICE), label.to(DEVICE)))
    model.to(DEVICE)

    start_time = datetime.now()
    input_size = (1, img_channel, img_size, img_size)
    random_input = torch.randn(input_size).to(DEVICE)
    layer_size_dict = utils.get_layer_inout_sizes(model, random_input)
    # layer_size_dict = utils.get_layer_output_sizes(model, random_input)

    cov = coverage.CGRC(model, 10, layer_size_dict,
                        p_value=0.05, chi2_test_threshold=5,
                        is_naive=False)

    # cov=coverage.NC(model, 0, layer_size_dict,)
    # cov=coverage.NCS(model, 0.5, layer_size_dict,)
    # cov = coverage.KMNC(model, 100, layer_size_dict,)
    # cov = coverage.NBC(model, 10, layer_size_dict,)
    # cov = coverage.SNAC(model, 10, layer_size_dict,)
    # cov = coverage.TKNC(model, 10, layer_size_dict,)
    # cov = coverage.TKNP(model, 10, layer_size_dict,)
    # cov = coverage.LSA(model, 0.001, 0.01, 10, layer_size_dict,) # threshold is the bucket size, to covert to percentage,
    # manually set max SA and n
    # cov = coverage.MDSA(model, 0.001, 0.01, 10, layer_size_dict,)
    # cov =coverage.DSA(model, 0.001, 0.01, 10, layer_size_dict,)

    # Update with Data List
    cov.build(data_list[:10])
    print(cov.current)

    # for i, (data, label) in enumerate(tqdm(data_list)):
    #     data = data.cuda()
    #     label = label.cuda()
    #     cov.build_step(data, label)
    # # [(batch_size, img_channel, img_size, img_size), ...]
    # print(cov.current)

    # Incremental Update
    # for data, label in data_list[10:]:
    # cov_dict = cov.calculate(data,label)
    for data in data_list[10:]:
        cov_dict = cov.calculate(data)
        inc = cov.gain(cov_dict)
        if inc is not None:
            cov.update(cov_dict, inc)
        print(cov.current)
    print(cov.current)
    print("Done!")
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    # # Save & Load
    # cov.save('/path/to/cov.pth')
    # cov.load('/path/to/cov.pth')
