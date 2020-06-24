import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from train_image import CUDA
from network import hook_w

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

CUDA = True


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def JKL(y_s, y_t):
    dkl_st = torch.sum(y_s * torch.log(y_s / (y_t + 1e-8)))
    dkl_ts = torch.sum(y_t * torch.log(y_t / y_s + 1e-8))
    return 0.5 * (dkl_st + dkl_ts)


def CDAN(input_list, ad_net,  w_s=None, w_t=None, random_layer=None):
    softmax_output = input_list[1].detach()
    features = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * features.size(1)))
    else:
        random_out = random_layer.forward([features, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if CUDA:
        dc_target = dc_target.cuda()

    if w_s is not None:
        source_weight = torch.zeros([2 * batch_size, 1]).cuda()
        target_weight = torch.zeros([2 * batch_size, 1]).cuda()
        source_weight[:features.size(0) // 2] = w_s
        target_weight[features.size(0) // 2:] = w_t

        weight = source_weight + target_weight
        weight = weight / weight.mean()
    else:
        weight = 1.

    return torch.mean(weight * nn.BCELoss(reduction='none')(ad_out, dc_target))


def DANN(features, ad_net, w_s=None, w_t=None, hook=True):
    ad_out = ad_net(features, hook=hook)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if CUDA:
        dc_target = dc_target.cuda()

    if CUDA:
        dc_target = dc_target.cuda()

    if w_s is not None:
        source_weight = torch.zeros([2 * batch_size, 1]).cuda()
        target_weight = torch.zeros([2 * batch_size, 1]).cuda()
        source_weight[:features.size(0) // 2] = w_s
        target_weight[features.size(0) // 2:] = w_t

        weight = source_weight + target_weight
        weight = weight / weight.mean()
    else:
        weight = 1.
    return torch.mean(weight * nn.BCELoss(reduction='none')(ad_out, dc_target))


def calc_temp(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0, temp_max=5., temp_min=1.):
    return (temp_max - temp_min)*(1 - np.float(2.0 * (high - low) /
                                               (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)) + temp_min


def RUDA(input_list, ad_net, w_s, w_t):
    softmax_output = input_list[1].detach()
    feature = input_list[0]

    ad_out = ad_net(feature)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array(np.ones([batch_size, softmax_output.shape[1]]).tolist() +
                                          np.zeros([batch_size, softmax_output.shape[1]]).tolist())).float()
    if CUDA:
        dc_target = dc_target.cuda()

    source_weight = torch.zeros([2 * batch_size, 1]).cuda()
    target_weight = torch.zeros([2 * batch_size, 1]).cuda()
    source_weight[:feature.size(0) // 2] = w_s
    target_weight[feature.size(0) // 2:] = w_t

    weight = source_weight + target_weight
    weight = weight / weight.mean()
    weight = torch.cat([weight for _ in range(softmax_output.shape[1])], dim=1)
    loss_w = (softmax_output * weight * nn.BCELoss(reduction='none')(ad_out, dc_target))
    return loss_w.mean(dim=0).sum()


def domain_accuracy(feature, output, ad_net, ad_w_net):
    ad_out_y = ad_w_net(feature, hook=False)
    ad_out_yf = ad_net(feature, hook=False)
    batch_size = ad_out_y.size(0) // 2

    weight_s, weight_t = w_from_ad(feature, ad_w_net, temp=1.)

    source_weight = torch.zeros([2 * batch_size, 1]).cuda()
    target_weight = torch.zeros([2 * batch_size, 1]).cuda()
    source_weight[:feature.size(0) // 2] = weight_s
    target_weight[feature.size(0) // 2:] = weight_t

    weight = source_weight + target_weight
    weight = weight / weight.mean()
    weight = torch.cat([weight for _ in range(output.shape[1])], dim=1)

    dc_target_y = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    dc_target_yf = torch.from_numpy(np.array(np.ones([batch_size, output.shape[1]]).tolist() +
                                             np.zeros([batch_size, output.shape[1]]).tolist())).float()

    if CUDA:
        dc_target_y = dc_target_y.cuda()
        dc_target_yf = dc_target_yf.cuda()

    acc_y = ((ad_out_y > 0.5).float() == dc_target_y).float().mean()
    acc_yf = (weight * output * ((ad_out_yf > 0.5).float() == dc_target_yf).float()).mean(dim=0).sum()
    return acc_y, acc_yf


def w_from_ad(feature, ad_w_net, temp, weight=True):
    ad_out = ad_w_net.forward(feature[:feature.size(0) // 2], temp, hook=True)
    eps = 1e-2
    w_s = (1. - ad_out) / (ad_out + eps)
    w_s = w_s / w_s.mean()

    if weight is False:
        return 0.*w_s + 1., 0. * w_s + 1.
    else:
        return w_s, 0. * w_s + 1.