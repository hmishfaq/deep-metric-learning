from __future__ import print_function
import random 
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init 
from torch.autograd import Variable
import numpy as np

import torchvision.models as models #



layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']




class VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self):
        super(VGG, self).__init__()

        #self.s = ngpu
        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            # if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            #     output = nn.parallel.data_parallel(
            #         module, output, range(self.ngpu))
            # else:
            output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs