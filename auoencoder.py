import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#download training and test data

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Create dataLoader
batch_size = 64
training_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

# for X,y in test_dataloader:
#     print(f"[N,C,H,W] {X.shape}")
#     print(f"taille , Type {y.shape},{y.dtype}")
#     break

# Create Model

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.flatten = nn.Flatten
        #28*28 = 784 ==> 128 ==> 64 ==> 9
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,9),
        )
        self.decoder = nn.Sequential(
          nn.Linear(9,64),
          nn.ReLU(),
          nn.Linear(64,128),
          nn.ReLU(),
          nn.Linear(128,784),
          nn.Sigmoid() 
        )

    def forward(self,x):
        x = self.flatten(x)
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder
    

