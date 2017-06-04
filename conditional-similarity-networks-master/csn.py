import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False,cuda = False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.cuda = cuda
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def reparametrize(self, mu, logvar,cuda = False):
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    #def decode(self,z):



    def forward(self, x, c):
        #embedded_x = self.embeddingnet(x)
        embedded_x_mean, embedded_x_logvar = self.embeddingnet(x) #VAE
        z = self.reparametrize(embedded_x_mean,embedded_x_logvar,self.cuda)

        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x_mean * self.mask
        return z,masked_embedding, self.mask.norm(1), embedded_x_mean.norm(2), masked_embedding.norm(2)