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
from models import wasserstein_loss
from utils import VisdomLinePlotter, VisdomImagePlotter


class Trainer():
    def __init__(
        self,
        device,
        GAN,
        dataloader,
        model_dir ='../models',
        num_epochs = 1,
        criterion = nn.BCELoss(),
        lr = 0.0002,
        beta1 = 0.5,
        nz = 100,
        real_label = 1.,
        fake_label = 0.,
        ):
        self.device=device
        self.GAN = GAN
        self.dataloader = dataloader
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.nz =nz
        self.fixed_noise=torch.randn(64, self.nz, 1, 1, device=self.device)
        self.real_label = real_label
        self.fake_label = fake_label
        self.optimizerD = optim.Adam(self.GAN.D.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.GAN.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.line_plotter = VisdomLinePlotter()
        self.image_plotter = VisdomImagePlotter()

    def train(self) :
        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.GAN.G(self.fixed_noise).detach().cpu()
                    self.image_plotter.plot(vutils.make_grid(fake, padding=2, normalize=True),name="generator-output")
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.GAN.D.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.GAN.D(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.GAN.G(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.GAN.D(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output,label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.GAN.G.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.GAN.D(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    self.line_plotter.plot(var_name = "loss", split_name= 'Discriminator', 
                    title_name = 'Training Loss', x = epoch + i/len(self.dataloader), y = errD.item())

                    self.line_plotter.plot(var_name = "loss", split_name= 'Generator', 
                    title_name = 'Training Loss', x = epoch + i/len(self.dataloader), y = errG.item())

                # # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.GAN.G(self.fixed_noise).detach().cpu()
                    self.image_plotter.plot(vutils.make_grid(fake, padding=2, normalize=True),name="generator-output")
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            with torch.no_grad():
                fake = self.GAN.G(self.fixed_noise).detach().cpu()
                self.image_plotter.plot(vutils.make_grid(fake, padding=2, normalize=True),name="generator-output")
                # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.GAN.G.state_dict(),
                        'optimizer_state_dict': self.optimizerG.state_dict(),
                        'loss': errG.item(),
                        }, os.path.join(self.model_dir,f"Generator-{epoch}.pth"))
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.GAN.D.state_dict(),
                        'optimizer_state_dict': self.optimizerD.state_dict(),
                        'loss': errD.item(),
                        }, os.path.join(self.model_dir,f"Discriminator-{epoch}.pth"))
