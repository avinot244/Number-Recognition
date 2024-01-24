import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from globals import *
from tqdm import tqdm

IMG_PATH_TRAIN = "../img/Perso/train/"

class ImgDataset(Dataset):
    def __init__(self, choice) -> None:
        super().__init__()
        self.x = []
        self.y = []

        if choice == "train":
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('C:/Users/Aymeric2/OneDrive/Documents/Loisir/Number recognition/nn/img/MNIST/train/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
            batch_size = batch_size_train, shuffle = True)
            for _ , (data, ground_truth) in enumerate(tqdm(train_loader, ncols=75)):
                for n in ground_truth:
                    if n == 0:
                        self.y.append(torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 1:
                        self.y.append(torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 2:
                        self.y.append(torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 3:
                        self.y.append(torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 4:
                        self.y.append(torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 5:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 6:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=torch.float32))
                    if n == 7:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=torch.float32))
                    if n == 8:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], dtype=torch.float32))
                    if n == 9:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32))

                for i in range(len(data)):
                    self.x.append(torch.reshape(data[i][0], (-1,)))

        if choice == "test":
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('C:/Users/Aymeric2/OneDrive/Documents/Loisir/Number recognition/nn/img/MNIST/test/', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=batch_size_test, shuffle=True)
            for _ , (data, ground_truth) in enumerate(tqdm(test_loader, ncols=75)):
                for n in ground_truth:
                    if n == 0:
                        self.y.append(torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 1:
                        self.y.append(torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 2:
                        self.y.append(torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 3:
                        self.y.append(torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 4:
                        self.y.append(torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 5:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=torch.float32))
                    if n == 6:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=torch.float32))
                    if n == 7:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=torch.float32))
                    if n == 8:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], dtype=torch.float32))
                    if n == 9:
                        self.y.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32))

                for i in range(len(data)):
                    self.x.append(torch.reshape(data[i][0], (-1,)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
def getData():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('C:/Users/Aymeric2/OneDrive/Documents/Loisir/Number recognition/nn/img/MNIST/train/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size = batch_size_train, shuffle = True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('C:/Users/Aymeric2/OneDrive/Documents/Loisir/Number recognition/nn/img/MNIST/test/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size_test, shuffle=True)
    return (train_loader, test_loader)