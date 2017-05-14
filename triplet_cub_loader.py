from __future__ import print_function
import torch.utils.data as data
import os
import errno
import torch
import json
import codecs
import numpy as np
import csv
import pandas as pd 
from PIL import Image

# README          bounding_boxes.txt  image_class_labels.txt  images.txt      shell_commands.txt  test_idx.txt        train_idx.txt
# attributes      classes.txt     images          parts           test_class_label.txt    train_class_label.txt   train_test_split.txt

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class CUB_t(data.Dataset):
    training_file = 'train_idx.txt'
    test_file = 'test_idx.txt'
    train_class_label_file ='train_class_label.txt'
    test_class_label_file ='test_class_label.txt'

    def __init__(self, root, n_train_triplets=50000, n_test_triplets=10000, num_classes=-1, train=True, transform=None, target_transform=None, download=False):

        self.loader = default_image_loader
        self.root = root
        
        self.transform = transform
        self.train = train  # training set or test set
        self.im_base_path = os.path.join(root, 'images')
        self.im_paths = pd.read_csv(os.path.join(root, 'images.txt'),
                                    names=['idx', 'path'], sep = " ")['path'].tolist()

        if num_classes < 0:
            self.num_classes = 200
        else:
            self.num_classes = min(num_classes, 200)

        if self.train:
            colnames = ['idx','labels']
            df = pd.read_csv(os.path.join(root, 'train_class_label.txt'),
                             names=colnames, sep = " ")
            self.train_idx = df['idx'].tolist()
            self.train_labels = df['labels'].tolist()
            self.triplets_train = self.make_triplet_list(n_train_triplets)

        else:
            colnames = ['idx','labels']
            df = pd.read_csv(os.path.join(root, 'test_class_label.txt'),
                             names=colnames, sep = " ")
            self.test_idx = df['idx'].tolist()
            self.test_labels = df['labels'].tolist()
            self.triplets_test = self.make_triplet_list(n_test_triplets)


    def __getitem__(self, index):
        if self.train:
            idx1, idx2, idx3 = self.triplets_train[index]
        else:
            idx1, idx2, idx3 = self.triplets_test[index]
        img1 = self.loader(os.path.join(self.im_base_path, self.im_paths[idx1]))
        img2 = self.loader(os.path.join(self.im_base_path, self.im_paths[idx2]))
        img3 = self.loader(os.path.join(self.im_base_path, self.im_paths[idx3]))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        if self.train:
            return len(self.triplets_train)
        else:
            return len(self.triplets_test)

    def make_triplet_list(self, ntriplets):
        print('Processing Triplet Generation ...')
        if self.train:
            np_labels = np.array(self.train_labels)
        else:
            np_labels = np.array(self.test_labels)
        triplets = []
        nc = int(self.num_classes)
        for class_idx in range(1,nc+1):
            # a, b, c are index of np_labels where it's equal to class_idx
            a = np.random.choice(np.where(np_labels==class_idx)[0],
                                 int(ntriplets/nc), replace=True)
            b = np.random.choice(np.where(np_labels==class_idx)[0],
                                 int(ntriplets/nc), replace=True)
            while np.any((a-b)==0): #aligning check. so that same indx at a and b wouldn't be same
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels!=class_idx)[0],
                                 int(ntriplets/nc), replace=True)

            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(c[i]), int(b[i])])
            for i in range(a.shape[0]):
                anchor,positive, negative =  int(a[i]), int(c[i]), int(b[i])
                if self.train:
                    triplets.append([self.train_idx[anchor],
                                     self.train_idx[positive],
                                     self.train_idx[negative]])          
                else:
                    triplets.append([self.test_idx[anchor],
                                     self.test_idx[positive],
                                     self.test_idx[negative]])          

        print('Done!')
        return triplets  # save the triplets to class
