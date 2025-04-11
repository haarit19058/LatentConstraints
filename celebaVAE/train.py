import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, sigma):
    std = sigma.exp()
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, n_latent):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 256, 5, 2), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2), nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2), nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, 2), nn.ReLU()
        )
        self.fc_mu = nn.Linear(2048 * 2 * 2, n_latent)
        self.fc_sigma = nn.Linear(2048 * 2 * 2, n_latent)

    def forward(self, x):
        h = self.conv_layers(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_sigma(h))
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, n_latent):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(n_latent, 2048 * 2 * 2)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, 2), nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, 2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 5, 2)
        )
    
    def forward(self, z):
        h = self.fc(z).view(-1, 2048, 2, 2)
        logits = self.deconv_layers(h)
        return logits

class Generator(nn.Module):
    def __init__(self, n_latent):
        super(Generator, self).__init__()
        self.fc_label = nn.Linear(10, 2048)
        self.fc_layers = nn.Sequential(
            nn.Linear(n_latent + 2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2 * n_latent)
        )
    
    def forward(self, z, labels):
        label_embedding = F.relu(self.fc_label(labels))
        x = torch.cat([z, label_embedding], dim=-1)
        x = self.fc_layers(x)
        dz, gates = x[:, :z.shape[1]], torch.sigmoid(x[:, z.shape[1]:])
        return (1 - gates) * z + gates * dz

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_label = nn.Linear(10, 2048)
        self.fc_layers = nn.Sequential(
            nn.Linear(2048 + 2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1)
        )
    
    def forward(self, z, labels):
        label_embedding = F.relu(self.fc_label(labels))
        x = torch.cat([z, label_embedding], dim=-1)
        return self.fc_layers(x)

class AttributeClassifier(nn.Module):
    def __init__(self, output_size=10):
        super(AttributeClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, output_size)
        )
    
    def forward(self, x):
        return self.fc_layers(x)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = Encoder(config['n_latent'])
        self.decoder = Decoder(config['n_latent'])
        self.generator = Generator(config['n_latent'])
        self.discriminator = Discriminator()
        self.attr_classifier = AttributeClassifier()
        self.n_latent = config['n_latent']
        self.x_sigma = config['x_sigma']
        self.beta = config['beta']
        self.lambda_weight = config['lambda_weight']
    
    def forward(self, x, labels):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        logits = self.decoder(z)
        x_recon = torch.sigmoid(logits)
        recons_loss = -torch.sum(F.mse_loss(x_recon, x, reduction='none'), dim=[1, 2, 3]).mean()
        kl_loss = 0.5 * torch.sum(mu ** 2 + sigma.exp() - sigma - 1, dim=-1).mean()
        vae_loss = -recons_loss + self.beta * kl_loss
        d_logits = self.discriminator(z, labels)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, torch.ones_like(d_logits))
        return vae_loss, d_loss



config = {
    'n_latent': 1024,
    'img_width': 64,
    'crop_width': 64,
    # Optimization parameters
    'batch_size': 128,
    'beta': 1.0,
    'x_sigma': 0.1,
    'lambda_weight': 10.0,
    'penalty_weight': 0.0,
}



n_latent = config['n_latent']

encoder = Encoder(n_latent)
classifier = AttributeClassifier()
decoder = Decoder(n_latent)


# Constants
transform_r_weight = torch.tensor(1.0)
transform_attr_weight = torch.tensor(0.0)
transform_penalty_weight = torch.tensor(0.0)
z_sigma_mean = torch.tensor(np.ones([1, n_latent]).astype(np.float32))

# Realism Constraint
loss_transform = transform_r_weight * torch.mean(d_loss)

# Attribute Constraint
loss_transform += transform_attr_weight * d_loss_attr

# Distance Penalty
transform_penalty = torch.log(1 + (z_prime - z0) ** 2)
transform_penalty = transform_penalty * z_sigma_mean**-2
loss_transform += torch.mean(transform_penalty_weight * transform_penalty)

# Amortized Transformation (Generator)
g_loss = -torch.log(torch.clamp(r_pred, 1e-15, 1 - 1e-15))
g_loss = torch.mean(g_loss)

g_penalty_weight = torch.tensor(0.0)
g_penalty = torch.log(1 + (z - q_z_sample) ** 2)
g_penalty = g_penalty * z_sigma_mean**-2
g_penalty = torch.mean(g_penalty)
g_loss += g_penalty_weight * g_penalty

# Classifier Loss
logits_classifier = classifier(x)
pred_classifier = torch.sigmoid(logits_classifier)
classifier_loss = nn.BCEWithLogitsLoss()(logits_classifier, labels)

# Learning Rates
d_lr = 3e-4
d_attr_lr = 3e-4
vae_lr = 3e-4
g_lr = 3e-4
classifier_lr = 3e-4
transform_lr = 3e-4

# Optimizers
vae_params = list(encoder.parameters()) + list(decoder.parameters())
train_vae = optim.Adam(vae_params, lr=vae_lr)

train_d = optim.Adam(d.parameters(), lr=d_lr, betas=(0, 0.9))
train_classifier = optim.Adam(classifier.parameters(), lr=classifier_lr)
train_g = optim.Adam(g.parameters(), lr=g_lr, betas=(0, 0.9))
train_d_attr = optim.Adam(d_attr.parameters(), lr=d_attr_lr)
train_transform = optim.Adam([z_prime], lr=transform_lr)

# Checkpoint Paths
basepath = os.path.expanduser('~/Desktop/CelebA/')
save_path = basepath

partition = np.loadtxt(basepath + 'list_eval_partition.txt', usecols=(1,))
train_mask = (partition == 0)
eval_mask = (partition == 1)
test_mask = (partition == 2)

print(f"Train: {train_mask.sum()}, Validation: {eval_mask.sum()}, Test: {test_mask.sum()}, Total: {partition.shape[0]}")











attributes = pd.read_table(basepath + 'list_attr_celeba.txt', skiprows=1, delim_whitespace=True, usecols=range(1, 41))
attribute_names = attributes.columns.values
attribute_values = attributes.values



attr_train = attribute_values[train_mask]
attr_eval = attribute_values[eval_mask]
attr_test = attribute_values[test_mask]

attr_train[attr_train == -1] = 0
attr_eval[attr_eval == -1] = 0
attr_test[attr_test == -1] = 0

np.save(basepath + 'attr_train.npy', attr_train)
np.save(basepath + 'attr_eval.npy', attr_eval)
np.save(basepath + 'attr_test.npy', attr_test)



def pil_crop_downsample(x, width, out_width):
  half_shape = tuple((i - width) / 2 for i in x.size)
  x = x.crop([half_shape[0], half_shape[1], half_shape[0] + width, half_shape[1] + width])
  return x.resize([out_width, out_width], resample=PIL.Image.ANTIALIAS)

def load_and_adjust_file(filename, width, outwidth):
  img = PIL.Image.open(filename)
  img = pil_crop_downsample(img, width, outwidth)
  img = np.array(img, np.float32) / 255.
  return img



# CELEBA images are (218 x 178) originally
filenames = np.sort(glob(basepath + 'img_align_celeba/*.jpg'))

crop_width = 128
img_width = 64
postfix = '_crop_%d_res_%d.npy' % (crop_width, img_width)

n_files = len(filenames)
all_data = np.zeros([n_files, img_width, img_width, 3], np.float32)
for i, fname in enumerate(filenames):
  all_data[i, :, :] = load_and_adjust_file(fname, crop_width, img_width)
  if i % 10000 == 0:
    print('%.2f percent done' % (float(i)/n_files * 100.0))
train_data = all_data[train_mask]
eval_data = all_data[eval_mask]
test_data = all_data[test_mask]
np.save(basepath + 'train' + postfix, train_data)
np.save(basepath + 'eval' + postfix, eval_data)
np.save(basepath + 'test' + postfix, test_data)












sess.run(tf.variables_initializer(var_list=m.vae_vars))

# Train the VAE
results = []
results_eval = []

traces = {'i': [],
          'i_eval': [],
          'loss': [],
          'loss_eval': [],
          'recons': [],
          'recons_eval': [],
          'kl': [],
          'kl_eval': []}

n_iters = 200000
vae_lr_ = np.logspace(np.log10(3e-4), np.log10(1e-6), n_iters)

for i in range(n_iters):
  start = (i * batch_size) % n_train
  end = start + batch_size
  batch = train_data[start:end]

  res = sess.run([m.train_vae,
                  m.vae_loss,
                  m.mean_recons,
                  m.mean_KL],
                 {m.x: batch,
                  m.vae_lr: vae_lr_[i],
                  m.amortize: False,
                  m.labels: attr_train[start:end]})

  traces['loss'].append(res[1])
  traces['recons'].append(res[2])
  traces['kl'].append(res[3])
  traces['i'].append(i)

  if i % 10 == 0:
    start = (i * batch_size) % n_eval
    end = start + batch_size
    batch = eval_data[start:end]
    res_eval = sess.run([m.vae_loss, m.mean_recons, m.mean_KL],
                        {m.x: batch, m.labels: attr_eval[start:end]})
    traces['loss_eval'].append(res_eval[0])
    traces['recons_eval'].append(res_eval[1])
    traces['kl_eval'].append(res_eval[2])
    traces['i_eval'].append(i)

    print('Step %5d \t TRAIN \t Loss: %0.3f, Recon: %0.3f, KL: %0.3f '
          '\t EVAL \t  Loss: %0.3f, Recon: %0.3f, KL: %0.3f' % (i,
                                                                rmean(traces['loss']),
                                                                rmean(traces['recons']),
                                                                rmean(traces['kl']),
                                                                rmeane(traces['loss_eval']),
                                                                rmeane(traces['recons_eval']),
                                                                rmeane(traces['kl_eval']) ))
    
    


plt.figure(figsize=(18,6))

plt.subplot(131)
plt.plot(traces['i'], traces['loss'])
plt.plot(traces['i_eval'], traces['loss_eval'])
plt.title('Loss')
# plt.ylim(30, 100)

plt.subplot(132)
plt.plot(traces['i'], traces['recons'])
plt.plot(traces['i_eval'], traces['recons_eval'])
plt.title('Recons')
# plt.ylim(-100, -30)

plt.subplot(133)
plt.plot(traces['i'], traces['kl'])
plt.plot(traces['i_eval'], traces['kl_eval'])
plt.title('KL')
# plt.ylim(10, 100)

