import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import network

CUDA = True


def train(args, i, model, ad_net, ad_w_net, train_loader, train_loader1, optimizer, optimizer_ad, optimizer_ad_w, epoch, start_epoch, method, random_layer=None):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target

    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        optimizer_ad_w.zero_grad()

        feature, output = model(torch.cat((data_source, data_target), 0))

        train_progress = (epoch - 1 + 1.*(batch_idx+1) / num_iter) / args.epochs

        temp = loss_func.calc_temp(train_progress, alpha=args.alpha, max_iter=1., temp_max=args.temp_max)
        w_s, w_t = loss_func.w_from_ad(feature, ad_w_net, temp=temp, weight=(args.weight == 1))

        w = torch.cat([w_s, w_t], dim=0)

        loss = (w_s.detach()*nn.CrossEntropyLoss(reduction='none')(output.narrow(0, 0, data_source.size(0)), label_source)).mean()
        softmax_output = nn.Softmax(dim=1)(output)

        if epoch > start_epoch:
            if method == 'CDAN':
                loss += loss_func.CDAN([feature, softmax_output], ad_net, w_s=w_s, w_t=w_t, random_layer=random_layer)
            elif method == 'DANN':
                loss += loss_func.DANN(feature, ad_net, w_s=w_s, w_t=w_t, hook=True)
            elif method == 'Y_DAN':
                loss += loss_func.Y_DAN([feature, softmax_output], ad_net, w_s=w_s, w_t=w_t)
            else:
                raise ValueError('Method cannot be recognized.')
        loss.backward(retain_graph=True)
        ad_w_net.zero_grad()
        optimizer.step()
        if epoch > start_epoch:
            optimizer_ad.step()

        invariance_loss = loss_func.DANN(feature.detach(), ad_w_net, hook=False)
        invariance_loss.backward()
        optimizer_ad_w.step()

        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
    parser.add_argument('--method', type=str, default='DANN')
    parser.add_argument('--task', default='MNIST2USPS', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str, default='0', help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False, help='whether to use random')
    parser.add_argument('--weight', type=int, default=0, help="whether use weights during transfer")
    parser.add_argument('--temp_max', type=float, default=5., help="weight relaxation parameter")
    parser.add_argument('--alpha', type=float, default=5., help="weight relaxation parameter")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.task == 'USPS2MNIST':
        source_list = '../data/usps2mnist/usps_train_shift_20.txt'
        target_list = '../data/usps2mnist/mnist_train.txt'
        test_list = '../data/usps2mnist/mnist_test.txt'
        start_epoch = 1
        decay_epoch = 6
    elif args.task == 'MNIST2USPS':
        source_list = '../data/usps2mnist/mnist_train_shift_20.txt'
        target_list = '../data/usps2mnist/usps_train.txt'
        test_list = '../data/usps2mnist/usps_test.txt'
        start_epoch = 1
        decay_epoch = 5
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    model = network.LeNet()
    model = model.cuda()
    class_num = 10

    if args.method is 'DANN':
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num(), 500)

    elif args.method is 'Y_DAN':
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num(), 500, output_dim=class_num)

    elif args.method is 'CDAN':
        if args.random:
            random_layer = network.RandomLayer([model.output_num(), class_num], 500)
            random_layer.cuda()
        else:
            random_layer = None
        ad_net = network.AdversarialNetwork(500*10, 500)
    else:
        raise ValueError('Method cannot be recognized.')

    ad_w_net = network.AdversarialNetwork(model.output_num(), 500, output_dim=1)

    ad_net = ad_net.cuda()
    ad_w_net = ad_w_net.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad_w = optim.SGD(ad_w_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        print("epoch", epoch)
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, epoch, model, ad_net, ad_w_net, train_loader, train_loader1,
              optimizer, optimizer_ad, optimizer_ad_w, epoch, start_epoch, args.method, random_layer=random_layer)
        test(args, model, test_loader)

if __name__ == '__main__':
    main()
