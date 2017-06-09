from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init #
import torchvision.models as models #

from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_image_loader import TripletImageLoader
#from tripletnet import CS_Tripletnet
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--vae_loss', type=float, default=1e-2, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
# parser.add_argument('--log_interval', type=int, default=20, metavar='N',
#                     help='how many batches to wait before logging training status')
# #####
# parser.add_argument('--image_size', type=int, default=112,
#                     help='height/width length of the input images, default=64')
parser.add_argument('--image_size', type=int, default=128, #112,
                    help='height/width length of the input images, default=64')

parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')

parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')

#same as dim_embed
parser.add_argument('--nz', type=int, default=64,
                    help='size of the latent vector z, default=64')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')


#####
parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0


#################################
#################################
#adapted from grabage.py :/
#these are layers of vgg19
layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']
###
content_layers = default_content_layers




#perceptual loss ? 

#################################
#################################

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


#####################
###
### Triplet Net
###
#####################


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        latent_x,mean_x,logvar_x = self.embeddingnet(x)
        latent_y,mean_y,logvar_y = self.embeddingnet(y)
        latent_z,mean_z,logvar_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(mean_x, mean_y, 2)
        dist_b = F.pairwise_distance(mean_x, mean_z, 2)
        return latent_x,mean_x,logvar_x,\
            latent_y,mean_y,logvar_y,\
            latent_z,mean_z,logvar_z,\
            dist_a, dist_b

#####################
###
### VGG Pretrained for perceptual loss
###
#####################
class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self, ngpu):
        super(_VGG, self).__init__()

        self.ngpu = ngpu
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
            if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(
                    module, output, range(self.ngpu))
            else:
                output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

#####################
###
### Encoder
###
#####################

class _Encoder(nn.Module):
    '''
    Encoder module, as described in :
    https://arxiv.org/abs/1610.00291
    Deep Feature Consistent Variational Autoencoder


    '''

    def __init__(self, ngpu,nc,nef,out_size,nz):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc 
        self.nef = nef
        self.out_size = out_size
        self.nz = nz
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            #Normalize(nef),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            # Normalize(nef * 2),
            nn.BatchNorm2d(nef*2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            # Normalize(nef * 4),
            nn.BatchNorm2d(nef*4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            # Normalize(nef * 8)
            nn.BatchNorm2d(nef*8),
        )
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar): #same as reparametrize(self, mu, logvar,cuda = False):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.encoder, input, range(self.ngpu))
            hidden = hidden.view(batch_size, -1)
            mean = nn.parallel.data_parallel(
                self.mean, hidden, range(self.ngpu))
            logvar = nn.parallel.data_parallel(
                self.logvar, hidden, range(self.ngpu))
        else:
            hidden = self.encoder(input)
            hidden = hidden.view(batch_size, -1)
            mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        #return latent_z
        return latent_z,mean,logvar

#####################
###
### Decoder
###
#####################


class _Decoder(nn.Module):
    '''
    Decoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu,nc,ndf,out_size,nz):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz  = nz
        self.ndf = ndf
        self.out_size = out_size
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * out_size * out_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf * 4, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf * 2, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.decoder_dense, input, range(self.ngpu))
            hidden = hidden.view(batch_size, self.ndf * 8, self.out_size, self.out_size)
            output = nn.parallel.data_parallel(
                self.decoder_conv, input, range(self.ngpu))
        else:
            hidden = self.decoder_dense(input).view(
                batch_size, self.ndf * 8, self.out_size, self.out_size)
            output = self.decoder_conv(hidden)
        return output



reconstruction_function = nn.BCELoss()
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD



# mse = nn.MSELoss()
# kld_criterion = nn.KLDivLoss()

# def fpl_criterion(recon_features, targets):
#     fpl = 0
#     for f, target in zip(recon_features, targets):
#         fpl += mse(f, target.detach())#.div(f.size(1))
#     return fpl




def train(train_loader, tnet,decoder, criterion, optimizer, epoch):
    losses_metric = AverageMeter()
    losses_VAE = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()


    # switch to train mode
    tnet.train()
    decoder.train()
    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)

        # compute output
        
        latent_x,mean_x,logvar_x,latent_y,mean_y,logvar_y,latent_z,mean_z,logvar_z,dist_a, dist_b = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        ### from ruishan###
        reconstructed_x = decoder(latent_x)
        reconstructed_y = decoder(latent_y)
        reconstructed_z = decoder(latent_z)
        loss_vae = loss_function(reconstructed_x, data1, mean_x, logvar_x)     
        loss_vae += loss_function(reconstructed_z, data2, mean_y, logvar_y)  
        loss_vae += loss_function(reconstructed_z, data3, mean_z, logvar_z)  
        loss_vae = loss_vae/(3*len(data1))

        #target - vec of 1. This is what i want : dista >distb = True
        loss_triplet = criterion(dist_a, dist_b, target)
        # loss_embedd = embed_norm / np.sqrt(data1.size(0))
        loss_embedd = mean_x.norm(2) + mean_y.norm(2) + mean_z.norm(2)
        
        loss = loss_triplet + args.embed_loss * loss_embedd + args.vae_loss*loss_vae
        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        losses_metric.update(loss_triplet.data[0], data1.size(0))
        losses_VAE.update(loss_vae.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # emb_norms.update(loss_embedd.data[0])
         
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'VAE Loss: {:.4f} ({:.4f}) \t'
                  'Metric Loss: {:.4f} ({:.4f}) \t'
                  'Metric Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses_VAE.val, losses_VAE.avg,
                losses_metric.val, losses_metric.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

            train_loss_metric.append(losses_metric.val)
            train_loss_VAE.append(losses_VAE.val)
            train_acc_metric.append(accs.val)


def test(test_loader, tnet, decoder,criterion, epoch):
    # losses = AverageMeter()
    # accs = AverageMeter()
    losses_metric = AverageMeter()
    accs = AverageMeter()

    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3,c) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3,c= data1.cuda(), data2.cuda(), data3.cuda(),c.cuda()
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)
        c_test = c 
         

        # compute output
        latent_x,mean_x,logvar_x,latent_y,mean_y,logvar_y,latent_z,mean_z,logvar_z,dist_a, dist_b = tnet(data1, data2, data3, c)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)

        reconstructed_x = decoder(latent_x)
        reconstructed_y = decoder(latent_y)
        reconstructed_z = decoder(latent_z)
        loss_vae = loss_function(reconstructed_x, data1, mean_x, logvar_x)     
        loss_vae += loss_function(reconstructed_z, data2, mean_y, logvar_y)  
        loss_vae += loss_function(reconstructed_z, data3, mean_z, logvar_z)  
        loss_vae = loss_vae/(3*len(data1))


        # test_loss =  criterion(dista, distb, target).data[0]
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = mean_x.norm(2) + mean_y.norm(2) + mean_z.norm(2)
        
        loss = loss_triplet + args.embed_loss * loss_embedd + args.vae_loss*loss_vae

        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        losses_metric.update(loss_triplet.data[0], data1.size(0))
        losses_VAE.update(loss_vae.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # emb_norms.update(loss_embedd.data[0])
    print('\nTest set: Average VAE loss: {:.4f}, Average Metric loss: {:.4f}, Metric Accuracy: {:.2f}%\n'.format(
            losses_VAE.avg, losses_metric.avg, 100. * accs.avg))
    test_loss_metric.append(losses_metric.avg)
    test_loss_VAE.append(losses_VAE.avg)
    test_acc_metric.append(accs.avg)   
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    # if args.visdom:
    #     plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(dist_a, dist_b):
    margin = 0
    pred = (dist_a - dist_b - margin).cpu().data
    return (pred > 0).sum()*1.0/dist_a.size()[0]

def accuracy_id(dist_a, dist_b, c, c_id):
    margin = 0
    pred = (dist_a - dist_b - margin).cpu().data
    return ((pred > 0)*(c.cpu().data == c_id)).sum()*1.0/(c.cpu().data == c_id).sum()


def main():
    global args, best_acc
    log_interval = 20
    global  log_interval
    args = parser.parse_args()
    print(args)
    nz = int(args.dim_embed)
    nef = int(args.nef)
    ndf = int(args.ndf)
    ngpu = int(args.ngpu)
    nc = 3
    out_size = args.image_size // 16
    # Normalize = nn.BatchNorm2d 

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #cuda = args.cuda
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # if args.visdom:
    #     global plotter 
    #     plotter = VisdomLinePlotter(env_name=args.name)
    
    #VGG!!!!
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # transform = transforms.Compose([
    #     transforms.Scale(args.image_size),
    #     #transforms.CenterCrop(args.image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                         std=(0.229, 0.224, 0.225))])

    global conditions
    #'Set of similarity notions'
    if args.conditions is not None:
        conditions = args.conditions
    else:
        conditions = [0,1,2,3]
    

    # out_size = args.image_size // 2**5 #5 is because in vgg19 there are 5 pool layers
    out_size = args.image_size // 16 # from garbage.py: Encoder makes x/2*4 w/ 4 conv/max layer
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'train', n_triplets=args.num_traintriplets,
                        transform=transforms.Compose([
                            transforms.Scale(args.image_size),
                            transforms.CenterCrop(args.image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'test', n_triplets=160000,
                        transform=transforms.Compose([
                            transforms.Scale(args.image_size),
                            transforms.CenterCrop(args.image_size),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'val', n_triplets=80000,
                        transform=transforms.Compose([
                            transforms.Scale(args.image_size),
                            transforms.CenterCrop(args.image_size),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    # model = Vgg_19.vgg19(pretrained=True, embedding_size=args.dim_embed,out_size =out_size)
    #embedding_size=64,output_size =8

    #### putting encoder instead of model
    encoder = _Encoder(ngpu,nc,nef,out_size,nz)
    encoder.apply(weights_init)
    # if args.encoder != '':
    #     encoder.load_state_dict(torch.load(args.encoder))
    print(encoder)
    if args.cuda:
        encoder = encoder.cuda()
    #changing csn input to encoder from model(vgg))
    tnet = Tripletnet(encoder)
    if args.cuda:
        tnet.cuda()

    decoder = _Decoder(ngpu,nc,ndf,out_size,nz)
    decoder.apply(weights_init)
    # if args.decoder != '':
    #     decoder.load_state_dict(torch.load(args.decoder))
    print(decoder)
     
    if args.cuda:
        decoder = decoder.cuda()

    # descriptor = _VGG(ngpu)
    # print(descriptor)
    # if args.cuda:
    #     descriptor = descriptor.cuda()

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

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    # parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    parameters = list(tnet.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params in tnet: {}'.format(n_parameters))


    #####
    #####
    global train_loss_metric,train_loss_VAE,train_acc_metric,test_loss_metric,test_loss_VAE,test_acc_metric
    train_loss_metric = []
    train_loss_VAE = []
    train_acc_metric = []
    test_loss_metric = []
    test_loss_VAE = []
    test_acc_metric = []
    ######
    if args.test:
        test_acc = test(test_loader, tnet,decoder, criterion, 1)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, tnet,decoder, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, tnet,decoder, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)



if __name__ == '__main__':
    main()    
