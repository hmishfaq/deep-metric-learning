from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tripletnet import Tripletnet
from visdom import Visdom
import numpy as np
from random import shuffle

"""
Base class for sampling.
"""
class Sampler(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    """
    Reset all samples, that is, clear all mined hard examples.
    """
    def Reset(self):
        pass

    """
    Call this after every batch to generate a list of hard negatives.
    """
    def SampleNegatives(self, dista, distb, triplet_loss, (idx1, idx2, idx3)):
        print("Implement me!!")
        pass

    """
    Call this after every batch to generate a list of hard positives.
    """
    def SamplePositives(self, dista, distb, triplet_loss, (idx1, idx2, idx3)):
        print("Implement me!!")
        pass

    """
    Call this when regenerating list of triplets, to get a set of negative pairs,
    or triplets with negative examples.
    """
    # TODO: may want to update the signature.
    def ChooseNegatives(self, num):
        print("Implement me!!")
        pass

    """
    Call this when regenerating list of triplets, to get a set of positive pairs,
    or triplets with positive examples.
    """
    # TODO: may want to update the signature.
    def ChoosePositives(self, num):
        print("Implement me!!")
        pass

"""
Get N hardest.
"""
class NHardestSampler(Sampler):
    def __init__(self, num_classes):
        super(NHardestSampler, self).__init__(num_classes)
        self.negatives = []  # list of anchor, negative pairs

    def Reset():
        self.negatives = []

    def SampleNegatives(self, dista, distb, triplet_loss, (idx1, idx2, idx3)):
        pass

