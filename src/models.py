import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


# Generator Code

class BaseGenerator(nn.Module):
    def __init__(self, ngpu = 1, nc =3,  ngf = 64 ,nz = 100,checkpoint_path=None):
        super(BaseGenerator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.checkpoint_path = checkpoint_path
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1 ,bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self._init_weights()

    def _init_weights(self):
        if self.checkpoint_path:
            print(f"loading weights from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif classname.find('BatchNorm') != -1:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        return self.main(input)

class BaseDiscriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf = 64,checkpoint_path=None):
        super(BaseDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.checkpoint_path = checkpoint_path
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0,bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        if self.checkpoint_path:
            print(f"loading weights from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif classname.find('BatchNorm') != -1:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, input):
        return self.main(input)

class BaseGan(nn.Module):
    def __init__(self, ngpu,device,G,D):
        super(BaseGan, self).__init__()
        self.ngpu=ngpu
        self.device = device
        self.G=G.to(device)
        self.D=D.to(device)
    
    def _init_weights(self):
        self.G._init_weights()
        self.D._init_weights()

    def forward(self,x):
        return x
    

def wasserstein_loss(d_real,d_fake):
    return -torch.mean(d_real) - torch.mean(d_fake)