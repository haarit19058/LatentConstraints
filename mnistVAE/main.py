import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs = 100
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

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

vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2).to(device)
optimizer = optim.Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)
            test_loss += loss_function(recon, data, mu, log_var).item()
    print(f'====> Test set loss: {test_loss / len(test_loader.dataset):.4f}')

for epoch in range(1, 51):
    train(epoch)
    test()

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

z_dim = 2
h_dim = 128
lr = 0.001
beta1 = 0.5

generator = Generator(z_dim, h_dim).to(device)
discriminator = Discriminator(z_dim, h_dim).to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()

num_epochs = 100
for epoch in range(num_epochs):
    x, y = [], []
    for data in train_loader:
        tempx, tempy = data[0].to(device), data[1].to(device)
        mask = (tempy == 7)
        x.append(tempx[mask])
    x = torch.cat(x).to(device)
    with torch.no_grad():
        mu, log_var = vae.encoder(x.view(-1, 784))
        z_real = vae.sampling(mu, log_var)
    z_fake = generator(torch.randn_like(z_real).to(device))
    real_labels = torch.ones(z_real.size(0), 1, device=device)
    fake_labels = torch.zeros(z_real.size(0), 1, device=device)
    optim_D.zero_grad()
    d_loss = criterion(discriminator(z_real), real_labels) + criterion(discriminator(z_fake.detach()), fake_labels)
    d_loss.backward()
    optim_D.step()
    optim_G.zero_grad()
    g_loss = criterion(discriminator(z_fake), real_labels)
    g_loss.backward()
    optim_G.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}]  D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

z_fake = generator(torch.randn((100,2), device=device))
with torch.no_grad():
    sample = vae.decoder(z_fake).to(device)

for image in sample.reshape(100,28,28):
    plt.imshow(image.cpu().numpy(), cmap='gray')
    plt.show()



torch.save(vae.state_dict(), "vae.pth")
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
