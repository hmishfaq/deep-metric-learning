from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import csv


class MNIST_regular(data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    train_triplet_file = 'train_triplets.txt'
    test_triplet_file = 'test_triplets.txt'

    def __init__(self, root,  n_train=60000, n_test=10000, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        
        self.n_train = n_train
        self.n_test = n_test
        
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        
        self.train_data, self.train_labels = torch.load(os.path.join(root, self.processed_folder, self.training_file))       
        self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))
        #self.train_data = self.train_data[:1000]
        #self.train_labels = self.train_labels[:1000]

        
    def __getitem__(self, index): 
        if self.train:
            class_idx = self.train_labels[index]
            img = self.train_data[index]
        else:
            class_idx = self.test_labels[index]
            img = self.test_data[index]
            
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, class_idx
        
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def getfeature(self):
       # test_idxs = np.random.randint(len(self.test_data), size = self.n_test)
       # train_idxs = np.random.randint(len(self.train_data), size = self.n_train)
        test_idxs = range(len(self.test_data))
        train_idxs = range(len(self.train_data))
        
        test_imgs = []
        test_classes = []
        train_imgs = []
        train_classes = []
        
        for index in train_idxs:
            class_idx = self.train_labels[index]
            img = self.train_data[index]
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            train_classes.append(class_idx)
            train_imgs.append(img)
              
        for index in test_idxs:
            class_idx = self.test_labels[index]
            img = self.test_data[index]
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            test_classes.append(class_idx)
            test_imgs.append(img)
            
        test_imgs = torch.stack(test_imgs)
        train_imgs = torch.stack(train_imgs)

        return test_imgs, test_classes, train_imgs, train_classes

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_triplets_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_triplet_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_triplet_file))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def make_triplet_list(self, ntriplets):

        if self._check_triplets_exists():
            return
        print('Processing Triplet Generation ...')
        if self.train:
            np_labels = self.train_labels.numpy()
            filename = self.train_triplet_file
        else:
            np_labels = self.test_labels.numpy()
            filename = self.test_triplet_file
        triplets = []
        for class_idx in range(10):
            a = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            b = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            while np.any((a-b)==0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels!=class_idx)[0], int(ntriplets/10), replace=True)

            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(c[i]), int(b[i])])           

        with open(os.path.join(self.root, self.processed_folder, filename), "wb") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(triplets)
        print('Done!')





def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)
