from __future__ import print_function
import random 
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init 
from torch.autograd import Variable
import numpy as np


nc = 3
# nef = 32 #int(args.nef)
Normalize = nn.BatchNorm2d
out_size = 64//16  #args.image_size // 16
nz = 100
ndf = 32

class Decoder(nn.Module):
    '''
    Decoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self):
        super(Decoder, self).__init__()
        # self.ngpu = ngpu
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * out_size * out_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 4, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 2, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     hidden = nn.parallel.data_parallel(
        #         self.decoder_dense, input, range(self.ngpu))
        #     hidden = hidden.view(batch_size, ndf * 8, out_size, out_size)
        #     output = nn.parallel.data_parallel(
        #         self.decoder_conv, input, range(self.ngpu))
        # else:
        hidden = self.decoder_dense(input).view(
            batch_size, ndf * 8, out_size, out_size)
        output = self.decoder_conv(hidden)
        return output





# class Decoder(nn.Module):
#     '''
#     Decoder module, as described in :
#     https://arxiv.org/abs/1610.00291
#     '''

#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.ngpu = ngpu
#         self.decoder_dense = nn.Sequential(
#             nn.Linear(nz, ndf * 8 * out_size * out_size),
#             nn.ReLU(True)
#         )
#         self.decoder_conv = nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             Normalize(ndf * 4, 1e-3),

#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             Normalize(ndf * 2, 1e-3),

#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(ndf * 2, ndf, 3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             Normalize(ndf, 1e-3),

#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(ndf, nc, 3, padding=1)
#         )

#     def forward(self, input):
#         batch_size = input.size(0)
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             hidden = nn.parallel.data_parallel(
#                 self.decoder_dense, input, range(self.ngpu))
#             hidden = hidden.view(batch_size, ndf * 8, out_size, out_size)
#             output = nn.parallel.data_parallel(
#                 self.decoder_conv, input, range(self.ngpu))
#         else:
#             hidden = self.decoder_dense(input).view(
#                 batch_size, ndf * 8, out_size, out_size)
#             output = self.decoder_conv(hidden)
#         return output