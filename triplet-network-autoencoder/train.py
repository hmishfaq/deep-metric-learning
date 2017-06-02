from __future__ import print_function
import argparse
import os
import random 
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_mnist_loader import MNIST_t
from triplet_image_loader import TripletImageLoader
from tripletnet import Tripletnet
from visdom import Visdom
import torchvision.models as models #

import numpy as np
from encoder import  Encoder
from decoder import Decoder
from vgg import VGG



layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']




# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

best_acc = 0
###########################
###########################
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent vector z, default=100')

parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')
# parser.add_argument('--ngpu', type=int, default=1,
#                     help='number of GPUs to use')
parser.add_argument('--image_size', type=int, default=64,
                    help='height/width length of the input images, default=64')

#print(ngpu)
###########################
###########################

def main():
    global args, best_acc
    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter 
    plotter = VisdomLinePlotter(env_name=args.name)
    ######################
    ##### NEW Transform for vgg19
    #Normalization mean and standard deviation are set accordingly to the ones used
    ######################
    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))])
    ######################
    ######################

    old_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]) 
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MNIST_t('../data', train=True, download=True,
                       transform=old_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MNIST_t('../data', train=False, transform=old_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # class Net(nn.Module):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #         self.conv2_drop = nn.Dropout2d()
    #         self.fc1 = nn.Linear(320, 50)
    #         self.fc2 = nn.Linear(50, 10)

    #     def forward(self, x):
    #         x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #         x = x.view(-1, 320)
    #         x = F.relu(self.fc1(x))
    #         x = F.dropout(x, training=self.training)
    #         return self.fc2(x)


    ######################
    ##### NEW
    ######################
    #ngpu = int(args.ngpu)

    nz = int(args.nz)
    nef = int(args.nef)
    ndf = int(args.ndf)
    nc = 3
    out_size = args.image_size // 16
    Normalize = nn.BatchNorm2d
    content_layers = default_content_layers
    ######################
    ######################
    descriptor = VGG()
    encoder = Encoder()
    #encoder.apply(weights_init)
    decoder = Decoder()
    #decoder.apply(weights_init)

    print(descriptor)
    model = encoder # Net()
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    ###
    mse = nn.MSELoss()
    kld_criterion = nn.KLDivLoss()
    ###
    criterion = torch.nn.MarginRankingLoss(margin = args.margin)

    #setup optimizer
    parameters = list(tnet.parameters()) + list(decoder.parameters())
    #optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
    #encoder param
    encoder_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of Encoder params: {}'.format(encoder_parameters))
    #decoder param
    decoder_parameters = sum([p.data.nelement() for p in decoder.parameters()])
    print('  + Number of decoder params: {}'.format(decoder_parameters))



    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, decoder,criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)














def train(train_loader, tnet,decoder, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    decoder.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)

        # These are feature output when data points are run through VGG19
        vgg_target_data1 = descriptor(data1)
        vgg_target_data2 = descriptor(data2)
        vgg_target_data3 = descriptor(data3)

        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        # KLD loss (need to check "latent_label")
        # kld = kld_criterion(F.log_softmax(latent_z), latent_labels)
        # kld.backward(retain_variables=True)

        #reconstruction errors for x, y, z
        recon_x,recon_y,recon_z = decoder(embedded_x),decoder(embedded_y),decoder(embedded_z)
        recon_x_features,recon_y_features,recon_z_features = descriptor(recon_x),descriptor(recon_y),descriptor(recon_z)
        fpl_x = fpl_criterion(recon_x_features, vgg_target_data1)
        fpl_y = fpl_criterion(recon_y_features, vgg_target_data2)
        fpl_z = fpl_criterion(recon_z_features, vgg_target_data3)
        fpl_x.backward()
        fpl_y.backward()
        fpl_z.backward()

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd + fpl_x+fpl_y+fpl_z

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    # log avg values to somewhere
    plotter.plot('acc', 'train', epoch, accs.avg)
    plotter.plot('loss', 'train', epoch, losses.avg)
    plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    plotter.plot('acc', 'test', epoch, accs.avg)
    plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

################################################
################################################
################################################
#
# NEW Functions based on  https://github.com/svenrdz/DFC-VAE/blob/master/main.py
#
################################################
################################################
################################################

def weights_init(m):
    '''
    Custom weights initialization called on encoder and decoder.
    '''
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()


def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())#.div(f.size(1))
    return fpl


if __name__ == '__main__':
    main()    
