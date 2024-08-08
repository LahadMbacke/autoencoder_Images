import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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


device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
# Create Model

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #28*28 = 784 ==> 128 ==> 64 ==> 9
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,9),
        )
        #9 ==> 64 ==> 128 ==> 784
        self.decoder = nn.Sequential(
          nn.Linear(9,64),
          nn.ReLU(),
          nn.Linear(64,128),
          nn.ReLU(),
          nn.Linear(128,784),
          nn.Sigmoid() 
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder
    

    
model = AutoEncoder().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)



def train_fn(training_dataloader,model,loss_fn,optimizer):
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_idx, (image, _) in enumerate(training_dataloader):
            image = image.view(-1, 28*28).to(device)
            predict = model(image)
            loss = loss_fn(predict, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:  # Afficher la perte tous les 100 batches
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

train_fn(training_dataloader,model,loss_fn,optimizer)

# Test and visualize 

def visualize_reconstruction(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            flattened_images = images.view(-1, 28*28)  # Aplatir les images
            reconstructed_images = model(flattened_images)
            reconstructed_images = reconstructed_images.view(-1, 1, 28, 28)  # Reformater pour affichage

            # Convertir les images pour affichage
            images = images.cpu()
            reconstructed_images = reconstructed_images.cpu()

            # Plot des images originales et reconstruites
            fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(10, 4))
            for i in range(8):
                axes[0, i].imshow(images[i].squeeze(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
                axes[1, i].axis('off')
            
            axes[0, 0].set_title('Original')
            axes[1, 0].set_title('Reconstructed')
            plt.show()
            break  # Affichez seulement le premier batch

visualize_reconstruction(test_dataloader, model, device)