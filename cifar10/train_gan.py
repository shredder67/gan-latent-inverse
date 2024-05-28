import argparse
import random
import pickle as pkl

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T
from torchvision.datasets import MNIST


class Generator(nn.Module):
    def __init__(self, ngpu=1, nc=1, nz=100, ngf=64):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
class Generator_CIFAR10(nn.Module):
    def __init__(self, ngpu=1, nc=1, nz=100, ngf=64):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

    
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
        
    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--outf', default='./models', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='manual seed')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    ### DATA PREPARATION  ###
    num_workers = args.workers
    batch_size = args.batch_size

    transform = T.Compose([
        T.Resize(28),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_data = MNIST(root='data', train=True,
                                    download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            num_workers=num_workers)
    #########################

    # input_size = 784
    # d_output_size = 1
    # d_hidden_size = 32

    # z_size = 100
    # g_output_size = 784
    # g_hidden_size = 32

    nz=100
    ngf=64
    ndf=64

    D = Discriminator(ndf=ndf).to(device)
    G = Generator(nz=nz, ngf=ngf).to(device)

    lr=0.0002
    beta1=0.5
    d_optimizer = optim.Adam(D.parameters(), lr, betas=(beta1, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr, betas=(beta1, 0.999))

    # training hyperparams
    num_epochs = 100

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []


    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, nz, 1, 1))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    D.train()
    G.train()

    load_bar = tqdm(range(num_epochs))
    save_img_during_training=True
    for epoch in load_bar:
        for batch_i, (real_images, _) in enumerate(train_loader):
                    
            batch_size = real_images.size(0)
            
            ## Important rescaling step ## 
            real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
            
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            
            d_optimizer.zero_grad()
            
            # 1. Train with real images

            # Compute the discriminator losses on real images 
            # smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)
            
            # 2. Train with fake images
            
            # Generate fake images
            # gradients don't have to flow during this step
            with torch.no_grad():
                z = np.random.uniform(-1, 1, size=(batch_size, nz, 1, 1))
                z = torch.from_numpy(z).float()
                fake_images = G(z)
            
            # Compute the discriminator losses on fake images        
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)
            
            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            
            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # 1. Train with fake images and flipped labels
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, nz, 1, 1))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake) # use real loss to flip labels
            
            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            load_bar.set_description('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        
        ## AFTER EACH EPOCH##
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to train mode

        if save_img_during_training:
            with open(f'logs/train_samples_e{epoch}.pkl', 'wb') as f:
                pkl.dump(samples[-1], f)


    # Save training generator samples
    with open('logs/train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # Save trained generator
    torch.save(G.state_dict(), 'models/mnist_gan.pth')