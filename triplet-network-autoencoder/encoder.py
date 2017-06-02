from __future__ import print_function
# import argparse
# import os
import random 
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init 
# import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
# from triplet_mnist_loader import MNIST_t
# from triplet_image_loader import TripletImageLoader
# from tripletnet import Tripletnet
import numpy as np



nc = 3
nef = 32 #int(args.nef)
Normalize = nn.BatchNorm2d
out_size = 64//16  #args.image_size // 16
nz = 100

class Encoder(nn.Module):
    '''
    Encoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self):
        super(Encoder, self).__init__()
        #self.ngpu = ngpu
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 8)
        )
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     hidden = nn.parallel.data_parallel(
        #         self.encoder, input, range(self.ngpu))
        #     hidden = hidden.view(batch_size, -1)
        #     mean = nn.parallel.data_parallel(
        #         self.mean, hidden, range(self.ngpu))
        #     logvar = nn.parallel.data_parallel(
        #         self.logvar, hidden, range(self.ngpu))
        # else:
        hidden = self.encoder(input)
        hidden = hidden.view(batch_size, -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z






# class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.conv2_drop = nn.Dropout2d()
#             self.fc1 = nn.Linear(320, 50)
#             self.fc2 = nn.Linear(50, 10)

#         def forward(self, x):
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, 320)
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             return self.fc2(x)


# class Encoder(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.conv2_drop = nn.Dropout2d()
#             self.fc1 = nn.Linear(320, 50)
#             self.fc2 = nn.Linear(50, 10)

#         def forward(self, x):
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, 320)
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             return self.fc2(x)