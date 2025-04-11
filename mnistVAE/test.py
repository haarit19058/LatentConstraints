import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchsummary import summary


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var



class Generator(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

class Discriminator(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))
    
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2).to(device)
# vae = VAE()


# Generator and discriminator dims
z_dim = 2
h_dim = 128
lr = 0.0002
beta1 = 0.5

generator = Generator(z_dim, h_dim).to(device)
discriminator = Discriminator(z_dim, h_dim).to(device)

vae.load_state_dict(torch.load("vae.pth", map_location=device))
generator.load_state_dict(torch.load("generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))

print(summary(vae,(1,28*28)))
print(summary(generator,(1,z_dim)))
print(summary(discriminator,(1,z_dim)))

vae.eval()
generator.eval()
discriminator.eval()



z_fake = generator(torch.randn((100, 2), device=device))
with torch.no_grad():
    sample = vae.decoder(z_fake).to(device)

sample = sample.reshape(100, 28, 28).cpu().numpy()

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample[i], cmap='gray')
    ax.axis("off")

plt.tight_layout()
plt.show()