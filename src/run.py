import models
import train
import torch
from data import Cifar10DataLoader, CustomDataLoader, ImageNetDataLoader
from pytorch_model_summary import summary


if __name__ == '__main__':
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    G = models.BaseGenerator(ngpu=1,ngf=64)
    D = models.BaseDiscriminator(ngpu=1,ndf=64)
    print(summary(G, torch.zeros((1,100, 1,1)), show_input=False))
    print(summary(D, torch.zeros((1, 3, 128,128)), show_input=False))
    # G = models.Generator(ngpu=1)
    # D = models.Discriminator(ngpu=1)
    GAN = models.BaseGan(ngpu=1,device=device,G=G,D=D)
    dataloader = CustomDataLoader(root="../datasets/coco",image_size=64).dataloader
    trainer = train.Trainer(device=device,GAN=GAN,dataloader=dataloader,num_epochs=20,model_dir="../models/Coco128")
    trainer.train()