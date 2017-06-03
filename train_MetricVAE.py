from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from triplet_mnist_loader import MNIST_t
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
        MNIST_t('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        MNIST_t('../data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MarginRankingLoss(margin = args.margin)


def train_test(epoch, trainlabel):
    losses_metric = AverageMeter()
    losses_VAE = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    
    if trainlabel == 'train':
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = test_loader
    
    for batch_idx, (data1, data2, data3) in enumerate(loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
              
        recon_batch1, mu1, logvar1 = model(data1)
        recon_batch2, mu2, logvar2 = model(data2)
        recon_batch3, mu3, logvar3 = model(data3)
        
        loss_vae = loss_function(recon_batch1, data1, mu1, logvar1)     
        loss_vae += loss_function(recon_batch2, data2, mu2, logvar2)  
        loss_vae += loss_function(recon_batch3, data3, mu3, logvar3)  
        loss_vae = loss_vae/(3*len(data1))
        
        dista = F.pairwise_distance(mu1, mu2, 2)
        distb = F.pairwise_distance(mu1, mu3, 2)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        loss_triplet = criterion(dista, distb, target)
        
        loss_embedd = mu1.norm(2) + mu2.norm(2) + mu3.norm(2)
        
        loss = 0.01*loss_vae + loss_triplet + 0.001*loss_embedd
        
        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses_metric.update(loss_triplet.data[0], data1.size(0))
        losses_VAE.update(loss_vae.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))
        
        # train
        if trainlabel == 'train':
            optimizer.zero_grad()          
            loss.backward()
            optimizer.step()  
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'VAE Loss: {:.4f} ({:.4f}) \t'
                      'Metric Loss: {:.4f} ({:.4f}) \t'
                      'Metric Acc: {:.2f}% ({:.2f}%) \t'
                      'Emb_Norm: {:.2f} ({:.2f})'.format(
                    epoch, batch_idx * len(data1), len(train_loader.dataset),
                    losses_VAE.val, losses_VAE.avg,
                    losses_metric.val, losses_metric.avg, 
                    100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    if trainlabel == 'test':
        print('\nTest set: Average VAE loss: {:.4f}, Average Metric loss: {:.4f}, Metric Accuracy: {:.2f}%\n'.format(
        losses_VAE.avg, losses_metric.avg, 100. * accs.avg))


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


for epoch in range(1, args.epochs + 1):
    train_test(epoch, 'train')
    train_test(epoch, 'test')

