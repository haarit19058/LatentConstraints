import torch
torch.cuda.set_device('cuda:0')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# pip install torch-summary

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch

# import torchsummary
import torchvision as tv
from torchvision import transforms, datasets
from torchvision.transforms import v2

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from torch.distributions.normal import Normal
import torch.nn.functional as F

import torch.nn as nn
from torch.nn import ReLU
from torch.optim.lr_scheduler import _LRScheduler

from time import time


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")











def create_filepaths():
    filenames = pd.read_csv('./data/list_eval_partition.csv')

    train_filenames = filenames[filenames['partition'] == 0]['image_id'].values
    val_filenames = filenames[filenames['partition'] == 1]['image_id'].values

    path_to_files = './data/img_align_celeba/img_align_celeba/'
    train_filepaths = path_to_files + train_filenames
    val_filepaths = path_to_files+val_filenames
    
    return train_filepaths, val_filepaths

def create_filepaths_all():
    filenames = pd.read_csv('./data/list_eval_partition.csv')

    imagename = filenames['image_id'].values
    # val_filenames = filenames[filenames['partition'] == 1]['image_id'].values

    path_to_files = './data/img_align_celeba/img_align_celeba/'
    # train_filepaths = path_to_files + train_filenames
    # val_filepaths = path_to_files+val_filenames
    
    filepaths = path_to_files + imagename
    
    return filepaths




# train_filepaths, val_filepaths = create_filepaths()

all_paths = create_filepaths_all()


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images[0].detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break



INPUT_SHAPE=(3,64,64)

class CreateDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = [img for img in imgs if self.is_valid(img)]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image = self.imgs[index]
        X = self.transform(image)
        return X, X
        
    def transform(self, path):
        img = tv.io.read_image(path)
        img = transforms.v2.functional.resized_crop(img, 40, 15, 148, 148, INPUT_SHAPE[1:], transforms.InterpolationMode.BILINEAR, True)/255.
        return img
    
    def is_valid(self, path):
        img = tv.io.read_image(path)
        return (img.shape[1] >= INPUT_SHAPE[1]) and (img.shape[2] >= INPUT_SHAPE[2])





# train_dataset = CreateDataset(train_filepaths)
# val_dataset = CreateDataset(val_filepaths)
dataset = CreateDataset(all_paths)

# train_dl = DataLoader(train_dataset, 32, shuffle=True, pin_memory=True, num_workers=2) #num_workers=3
# val_dl = DataLoader(val_dataset, 32, shuffle=True, pin_memory=True, num_workers=2)
dataloader = DataLoader(dataset, 32, shuffle=True, pin_memory=True, num_workers=2)


fig,axes = plt.subplots(1,2)
axes[0].imshow(tv.io.read_image(all_paths[0]).permute(1, 2, 0))
axes[0].set_title('Before transformation',fontsize=20)
axes[1].imshow(dataset.transform(all_paths[0]).permute(1,2,0))

axes[1].set_title('After transformation',fontsize=20)
plt.tight_layout()
plt.show()

# plt.figure()
# show_batch(train_dl, 32)
# plt.savefig("./images/batch.jpg", bbox_inches='tight')
# plt.close()









import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

#########################################
# CelebA Convolutional VAE Components
#########################################

class CelebAEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(CelebAEncoder, self).__init__()
        # Four 2D convolutional layers.
        self.conv1 = nn.Conv2d(3, 2048, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=2)
        # For a 64×64 input, after 4 conv layers the feature map becomes 4×4.
        self.fc = nn.Linear(256 * 4 * 4, 2048)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 2048, 32, 32]
        x = F.relu(self.conv2(x))   # [B, 1024, 16, 16]
        x = F.relu(self.conv3(x))   # [B, 512, 8, 8]
        x = F.relu(self.conv4(x))   # [B, 256, 4, 4]
        x = x.view(x.size(0), -1)     # Flatten
        x = self.fc(x)              # [B, 2048]
        # Split into two halves: one for μ and one for log-σ (pre-softplus)
        mu, log_sigma = torch.chunk(x, 2, dim=1)
        # Ensure σ > 0 using softplus.
        sigma = F.softplus(log_sigma)
        return mu, sigma

class CelebADecoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(CelebADecoder, self).__init__()
        # Transform the latent vector into a seed feature map.
        self.fc = nn.Linear(latent_dim, 2048 * 4 * 4)
        # Four transposed convolutional layers.
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2,
                                          padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2,
                                          padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 2048, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # Use sigmoid to map the output to [0, 1].
        x = torch.sigmoid(self.deconv4(x))
        return x

class CelebAVAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(CelebAVAE, self).__init__()
        self.encoder = CelebAEncoder(latent_dim)
        self.decoder = CelebADecoder(latent_dim)
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        recon = self.decoder(z)
        return recon, mu, sigma

#########################################
# Loss Function for the VAE
#########################################

def loss_function(recon_x, x, mu, sigma):
    # Reconstruction loss (BCE) summed over all pixels.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence between the approximate posterior and the unit Gaussian.
    KL = -0.5 * torch.sum(1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2))
    return BCE + KL

#########################################
# Training the CelebA VAE
#########################################


# Hyperparameters
batch_size = 64
epochs = 20
latent_dim = 1024
learning_rate = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Data transforms: center-crop to 128×128, then resize to 64×64.
# transform = transforms.Compose([
#     transforms.CenterCrop(128),
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),  # Scales images to [0,1]
# ])

# # Load the CelebA training split (download if needed).
# dataset = datasets.CelebA(root='./data3', split='train', download=False, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model and optimizer.
model = CelebAVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

model.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, sigma = model(data)
        loss = loss_function(recon, data, mu, sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            avg_batch_loss = loss.item() / data.size(0)
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {avg_batch_loss:.4f}")
    
    avg_loss = train_loss / len(dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    
    # Optionally, save a grid of reconstructed images for visual monitoring.
    with torch.no_grad():
        sample_batch = next(iter(dataloader))[0].to(device)
        recon_sample, _, _ = model(sample_batch)
        grid = utils.make_grid(recon_sample[:16], nrow=4)
        os.makedirs('results', exist_ok=True)
        utils.save_image(grid, f"results/epoch_{epoch}.png")

    # Save the model checkpoint.
    torch.save(model.state_dict(), "celeba_vae.pth")
print("Training finished and model saved.")




