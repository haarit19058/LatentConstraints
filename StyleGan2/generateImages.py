import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import dnnlib
import legacy
from torchvision.utils import save_image
from torchvision import models

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained StyleGAN2 model
network_pkl = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl"
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
G.eval()
for param in G.parameters():
    param.requires_grad = False

# Ensure the output folder exists
output_folder = "../styleGanImages"
os.makedirs(output_folder, exist_ok=True)

num_images = 500000
batch_size = 32  # Adjust based on your GPU memory
num_batches = num_images // batch_size

start_time = time.time()
latent_vectors = []
with torch.no_grad():
    for batch_idx in range(num_batches):
        # Generate batch of latent vectors
        z = torch.randn(batch_size, 512, device=device)
        latent_vectors.append(z)
        # Mapping network
        w = G.mapping(z, None)
        # Synthesis network to generate images
        generated_images = G.synthesis(w, noise_mode='const')

        # Process and save each image in the batch
        for i in range(batch_size):
            img = (generated_images[i].clamp(-1, 1) + 1) / 2 * 255
            img = img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            filename = os.path.join(output_folder, f"image_{batch_idx * batch_size + i:09d}.png")
            plt.imsave(filename, img)

        # Optionally print progress every 100 batches
        if (batch_idx + 1) % 1000 == 0:
            print(f"{(batch_idx + 1) * batch_size} images generated.")
            latent_vectors = np.array(latent_vectors)
            latent_vectors_file = "latent_vectors.npy"
            np.save(latent_vectors_file, latent_vectors)
            print(f"Latent vectors saved to {latent_vectors_file}")
            latent_vectors = list(latent_vectors)

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / num_images

print(f"Finished generating {num_images} images.")
print(f"Time taken: {total_time} seconds")
print(f"Average time per image: {avg_time} seconds")


# Log the performance metrics
with open("log.txt", "w") as f:
    f.write(f"Time taken to generate {num_images} images: {total_time} seconds\n")
    f.write(f"Average time per image: {avg_time} seconds\n")
    f.write(f"Total images generated: {num_images}\n")
