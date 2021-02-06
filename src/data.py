import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


class DataSetLoader():
    def __init__(
        self,
        batch_size = 128,
        shuffle = True,
        num_workers = 2

    ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


class Cifar10DataLoader(DataSetLoader):
    def __init__(
        self,
        image_size,
        root ="../datasets/cifar10",
        

    ):
        super(Cifar10DataLoader,self).__init__() 
        self.root = root
        self.image_size = image_size
        self.dataset = dset.CIFAR10(root="../datasets/cifar10",
                                transform=transforms.Compose([
                                    transforms.Resize(self.image_size),
                                    transforms.CenterCrop(self.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                shuffle=self.shuffle, num_workers = self.num_workers)

class ImageNetDataLoader(DataSetLoader):
    def __init__(
        self,
        image_size,
        root ="../datasets/Imagenet"
    ):
        super(ImageNetDataLoader,self).__init__() 
        self.root = root
        self.image_size = image_size
        self.dataset = dset.ImageNet(root="../datasets/Imagenet",
                                    download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(self.image_size),
                                    transforms.CenterCrop(self.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                shuffle=self.shuffle, num_workers = self.num_workers)

class CustomDataLoader(DataSetLoader):
    def __init__(
        self,
        image_size,
        root ="../datasets/custom"
    ):
        super(CustomDataLoader,self).__init__() 
        self.root = root
        self.image_size = image_size
        self.dataset = dset.ImageFolder(root=self.root,
                                transform=transforms.Compose([
                                    transforms.Resize(self.image_size),
                                    transforms.CenterCrop(self.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                shuffle=self.shuffle, num_workers = self.num_workers)