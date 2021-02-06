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
import models
from utils import VisdomImagePlotter



def make_inference(model_name='Generator',checkpoint_path=None):
    image_plotter = VisdomImagePlotter(env_name="test")
    ngpu=1
    b_size=128
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = models.Generator(checkpoint_path=checkpoint_path)
    model.to(device)
    noise = torch.randn(b_size, model.nz, 1, 1, device=device)
    with torch.no_grad():
        fake = model(noise).detach().cpu()
        image_plotter.plot(vutils.make_grid(fake, padding=2, normalize=True),name="generator-output")


if __name__ == '__main__':
    make_inference(checkpoint_path="../models/Generator-9.pth")
