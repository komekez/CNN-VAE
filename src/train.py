import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from engine import train, validate
from utils import saveReconstructedImage, imageToVid, saveLossPlot

matplotlib.style.use('ggplot')


model = model.convVAE()
lr=0.001
epochs = 100
batch_size=64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(
    root='../input', train=True, download=True, transform=transform
)

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../input', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, trainloader, trainset, criterion)
    train_loss.append(train_epoch_loss)

    val_epoch_loss, recon_image = validate(model, testloader, testset, criterion)
    val_loss.append(val_epoch_loss)

    saveReconstructedImage(recon_image, epoch+1)
    image_grid = make_grid(recon_image.detach())

    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

    # save the reconstructions as a .gif file
    imageToVid(grid_images)
    # save the loss plots to disk
    saveLossPlot(train_loss, val_loss)
    print('TRAINING COMPLETE')



