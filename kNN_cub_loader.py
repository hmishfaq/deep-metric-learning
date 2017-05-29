from __future__ import print_function
import torch.utils.data as data
import os
import math
import errno
import torch
import json
import codecs
import numpy as np
import csv
import pandas as pd 
from PIL import Image
import hard_mining

# README          bounding_boxes.txt  image_class_labels.txt  images.txt      shell_commands.txt  test_idx.txt        train_idx.txt
# attributes      classes.txt     images          parts           test_class_label.txt    train_class_label.txt   train_test_split.txt

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class CUB_t_kNN(data.Dataset):
    training_file = 'train_idx.txt'
    test_file = 'test_idx.txt'
    train_class_label_file ='train_class_label.txt'
    test_class_label_file ='test_class_label.txt'

    def __init__(self, root, n_train=100, n_test=100, num_classes=-1, train=True, transform=None, target_transform=None, download=False):

        self.loader = default_image_loader
        self.root = root
        
        self.n_test = n_test
        self.n_train = n_train
        
        self.transform = transform
        self.train = train  # training set or test set
        self.im_base_path = os.path.join(root, 'images')
        self.im_paths = pd.read_csv(os.path.join(root, 'images.txt'),
                                    names=['idx', 'path'], sep = " ")['path'].tolist()

        if num_classes < 0:
            self.num_classes = 200
        else:
            self.num_classes = min(num_classes, 200)

        # train
        colnames = ['idx','labels']
        df = pd.read_csv(os.path.join(root, 'train_class_label.txt'),
                         names=colnames, sep = " ")
        self.train_idx = df['idx'].tolist()
        self.train_labels = df['labels'].tolist()

        # test
        colnames = ['idx','labels']
        df = pd.read_csv(os.path.join(root, 'test_class_label.txt'),
                         names=colnames, sep = " ")
        self.test_idx = df['idx'].tolist()
        self.test_labels = df['labels'].tolist()


    def getitem(self):
        test_idxs = np.random.randint(len(self.test_idx), size = self.n_test)
        train_idxs = np.random.randint(len(self.train_idx), size = self.n_train)
        
        test_imgs = []
        test_classes = []
        train_imgs = []
        train_classes = []
        
        for index in test_idxs:
            idx = self.test_idx[index]
            class_idx = self.test_labels[index]
            img = self.loader(os.path.join(self.im_base_path, self.im_paths[idx]))
            img = img.resize((64,64))
            if self.transform is not None:
                img = self.transform(img)
            test_imgs.append(img)
            test_classes.append(class_idx)
            
        for index in train_idxs:
            idx = self.train_idx[index]
            class_idx = self.train_labels[index]
            img = self.loader(os.path.join(self.im_base_path, self.im_paths[idx]))
            img = img.resize((64,64))
            if self.transform is not None:
                img = self.transform(img)
            train_imgs.append(img)
            train_classes.append(class_idx)
            
        test_imgs = torch.stack(test_imgs)
        train_imgs = torch.stack(train_imgs)

        return test_imgs, test_classes, train_imgs, train_classes

