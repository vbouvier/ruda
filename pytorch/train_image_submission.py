####################################################################################
# We thank MingSheng Long et. al.; their implementation has been used as a basis.
# Paper: https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation
# Code: https://github.com/thuml/CDAN
####################################################################################

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import *
from loss import *
from pre_process import *
from torch.utils.data import DataLoader
from lr_schedule import *
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

CUDA = True


def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    if CUDA:
                        inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
            del iter_test, data, inputs, labels
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                if CUDA:
                    inputs = inputs.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    del predict, all_output, all_label
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = image_train(**config["prep"]['params'])
    prep_dict["target"] = image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    if CUDA:
        base_network = base_network.cuda()

    ## add additional network for some methods

    if config['method'] is 'DANN':
        ad_net = AdversarialNetwork(base_network.output_num(), 1024)
    elif config['method'] is 'RUDA':
        ad_net = AdversarialNetwork(base_network.output_num(), 1024,
                                    output_dim=config["network"]["params"]["class_num"])
    elif config['method'] is 'CDAN':
        random_layer = RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        if CUDA:
            random_layer.cuda()
        ad_net = AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        print(config['method'])
        raise ValueError('Method cannot be recognized.')

    ad_w_net = AdversarialNetwork(base_network.output_num(), 1024, output_dim=1)

    if CUDA:
        ad_net = ad_net.cuda()
        ad_w_net = ad_w_net.cuda()


    parameter_list = base_network.get_parameters() + ad_net.get_parameters() + ad_w_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        ad_w_net = nn.DataParallel(ad_w_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    best_acc = 0.0


    for i in tqdm(range(config["num_iterations"])):
        if i > config['start_test']:
            if i % config["test_interval"] == config["test_interval"] - 1:

                base_network.train(False)

                temp_acc = image_classification_test(dset_loaders, base_network, test_10crop=prep_config["test_10crop"])

                if temp_acc > best_acc:
                    best_acc = temp_acc
                log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
                print(log_str)
                print(calc_coeff(i))
                print('Temp', calc_temp(i, alpha=config['alpha']))
                print('Weigth std source', np.mean(w_s_std))

        loss_params = config["loss"]

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        ad_w_net.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)

        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()

        if CUDA:
            inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        temp = calc_temp(i, alpha=config['alpha'])

        w_s, w_t = w_from_ad(features, ad_w_net, temp=temp, weight=config['weight'])

        invariance_loss = DANN(features.detach(), ad_w_net, hook=False)

        if config['method'] is 'DANN':
            transfer_loss = DANN(features, ad_net, w_s, w_t, hook=True)
        elif config['method'] is 'CDAN':
            transfer_loss = CDAN([features, softmax_out], ad_net, w_s, w_t, random_layer=random_layer)
        elif config['method'] is 'RUDA':
            transfer_loss = RUDA([features, softmax_out], ad_net, w_s, w_t)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = (w_s.detach()*nn.CrossEntropyLoss(reduction='none')(outputs_source, labels_source)).mean()

        (classifier_loss + loss_params["trade_off"]*transfer_loss).backward(retain_graph=True)
        ad_w_net.zero_grad()
        invariance_loss.backward()
        optimizer.step()
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robust Domain Adaptation (RUDA)')
    parser.add_argument('--method', type=int, default=2, choices=[0, 1, 2], help='Method 0: DANN, 1: CDAN, 2: RUDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office_shift', help="The dataset or source dataset used")
    parser.add_argument('--tag', type=str, default='D_A', help="Tag of the experiments for saving")
    parser.add_argument('--test_interval', type=int, default=1000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--random', type=bool, default=True, help="whether use random projection")
    parser.add_argument('--weight', type=int, default=1, help="whether use weights during transfer")
    parser.add_argument('--alpha', type=float, default=5., help="weight relaxation parameter")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    methods = {0: 'DANN', 1: 'CDAN', 2: 'RUDA'}

    # train config
    config = {}
    config['method'] = methods[args.method]
    config['weight'] = args.weight == 1
    config['alpha'] = args.alpha
    config['tag'] = config['method'] + '_' + str(config['weight']) + '_' + str(args.alpha) + '_' + args.tag + '_' + str(args.seed)

    print('Adaptation starts with ' + config['method'] + '/ Uses weights: ' + config['weight'])
    print('Logging done at ', config['tag'])

    config['start_test'] = 1

    if args.dset == 'office':
        config['tag'] += '_standard'
        if args.tag == 'A_W':
            s_dset_path = '../data/office/amazon_list.txt'
            t_dset_path = '../data/office/webcam_list.txt'

        elif args.tag == 'W_A':
            s_dset_path = '../data/office/webcam_list.txt'
            t_dset_path = '../data/office/amazon_list.txt'

        elif args.tag == 'A_D':
            s_dset_path = '../data/office/amazon_list.txt'
            t_dset_path = '../data/office/dslr_list.txt'

        elif args.tag == 'D_A':
            s_dset_path = '../data/office/dslr_list.txt'
            t_dset_path = '../data/office/amazon_list.txt'

        elif args.tag == 'D_W':
            s_dset_path = '../data/office/dslr_list.txt'
            t_dset_path = '../data/office/webcam_list.txt'

        elif args.tag == 'W_D':
            s_dset_path = '../data/office/webcam_list.txt'
            t_dset_path = '../data/office/dslr_list.txt'

    elif args.dset == 'office_shift':
        config['tag'] += '_shift'
        if args.tag == 'A_W':
            s_dset_path = '../data/office/amazon_list_shift_5.txt'
            t_dset_path = '../data/office/webcam_list.txt'

        elif args.tag == 'W_A':
            s_dset_path = '../data/office/webcam_list_shift_5.txt'
            t_dset_path = '../data/office/amazon_list.txt'

        elif args.tag == 'A_D':
            s_dset_path = '../data/office/amazon_list_shift_5.txt'
            t_dset_path = '../data/office/dslr_list_local.txt'

        elif args.tag == 'D_A':
            s_dset_path = '../data/office/dslr_list_shift_5.txt'
            t_dset_path = '../data/office/amazon_local.txt'

        elif args.tag == 'D_W':
            s_dset_path = '../data/office/dslr_list_shift_5.txt'
            t_dset_path = '../data/office/webcam_local.txt'

        elif args.tag == 'W_D':
            s_dset_path = '../data/office/webcam_list_shift_5.txt'
            t_dset_path = '../data/office/dslr_list_local.txt'

    elif args.dset == 'office-home':
        config['tag'] += '_standard'
        if args.tag == 'A_C':
            s_dset_path = '../data/office-home/Art.txt'
            t_dset_path = '../data/office-home/Clipart.txt'
        elif args.tag == 'A_P':
            s_dset_path = '../data/office-home/Art.txt'
            t_dset_path = '../data/office-home/Product.txt'
        elif args.tag == 'A_P':
            s_dset_path = '../data/office-home/Art.txt'
            t_dset_path = '../data/office-home/Product.txt'
        elif args.tag == 'A_R':
            s_dset_path = '../data/office-home/Art.txt'
            t_dset_path = '../data/office-home/Real_World.txt'
        elif args.tag == 'C_A':
            s_dset_path = '../data/office-home/Clipart.txt'
            t_dset_path = '../data/office-home/Art.txt'
        elif args.tag == 'C_P':
            s_dset_path = '../data/office-home/Clipart.txt'
            t_dset_path = '../data/office-home/Product.txt'
        elif args.tag == 'C_R':
            s_dset_path = '../data/office-home/Clipart.txt'
            t_dset_path = '../data/office-home/Real_World.txt'
        elif args.tag == 'P_A':
            s_dset_path = '../data/office-home/Product.txt'
            t_dset_path = '../data/office-home/Art.txt'
        elif args.tag == 'P_C':
            s_dset_path = '../data/office-home/Product.txt'
            t_dset_path = '../data/office-home/Clipart.txt'
        elif args.tag == 'P_R':
            s_dset_path = '../data/office-home/Product.txt'
            t_dset_path = '../data/office-home/Real_World.txt'
        elif args.tag == 'R_A':
            s_dset_path = '../data/office-home/Real_World.txt'
            t_dset_path = '../data/office-home/Art.txt'
        elif args.tag == 'R_C':
            s_dset_path = '../data/office-home/Real_World.txt'
            t_dset_path = '../data/office-home/Clipart.txt'
        elif args.tag == 'R_P':
            s_dset_path = '../data/office-home/Real_World.txt'
            t_dset_path = '../data/office-home/Product.txt'

    config["gpu"] = args.gpu_id
    config["num_iterations"] = 20004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_dir"] = config['tag']
    config["output_path"] = "snapshot/" + config["output_dir"]
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path": s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path": t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":t_dset_path, "batch_size":4}}

    if ("office" == config["dataset"]) or ("office_shift" == config["dataset"]):
        if ("amazon" in s_dset_path and "webcam" in t_dset_path) or \
           ("webcam" in s_dset_path and "dslr" in t_dset_path) or \
           ("webcam" in s_dset_path and "amazon" in t_dset_path) or \
           ("dslr" in s_dset_path and "amazon" in t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in s_dset_path and "dslr" in t_dset_path) or \
             ("dslr" in s_dset_path and "webcam" in t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
