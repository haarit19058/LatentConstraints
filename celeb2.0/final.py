import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import logging
import time
import torchvision as tv
from torchvision import transforms
from torchvision.transforms import v2  # Note: Ensure torchvision version supports v2
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler # Although not used in the original training loop provided

# ==================================
# 1. Configuration & Setup
# ==================================

# --- Directories ---
OUTPUT_DIR = 'final_output'
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'actor_critic_models')
DATA_DIR = './data'
ATTR_CSV_PATH = os.path.join(DATA_DIR, 'list_attr_celeba.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'img_align_celeba/img_align_celeba')
VAE_MODEL_PATH = 'celeba_vae.pth' # Assuming VAE model is in the root directory

# --- Device ---
if torch.cuda.is_available():
    # Explicitly set device if needed, otherwise let PyTorch choose
    # torch.cuda.set_device('cuda:0') # Uncomment if specific GPU is required
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Hyperparameters ---
LATENT_DIM = 1024
ATTR_DIM = 10 # Number of selected attributes
VAE_TRAIN_TIME = None # Placeholder, VAE is pre-trained in this script
GENERATOR_TRAIN_TIME = None # Will be calculated

# Training Hyperparameters for Actor/Critic
AC_BATCH_SIZE = 64 # Actor-Critic training batch size
AC_EPOCHS = 10 # Reduced for quick testing, original was 1000
AC_LEARNING_RATE = 0.0001
AC_LAMBDA_DIST = 0.1 # Weight for the distance penalty in generator loss

# Dataset Hyperparameters
DS_BATCH_SIZE = 32  # Batch size for creating the latent dataset
DS_SAMPLES_PER_IMAGE = 10 # Reduced for faster dataset creation, original was 100

# --- Output Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Logging Setup ---
log_file = os.path.join(OUTPUT_DIR, 'training_log.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Log to console as well
    ]
)
logging.info("=" * 40)
logging.info("Script Execution Started")
logging.info(f"Output Directory: {OUTPUT_DIR}")
logging.info(f"Using device: {DEVICE}")
logging.info(f"Pre-trained VAE Path: {VAE_MODEL_PATH}")


# ==================================
# 2. Model Definitions (VAE, Actor, Critic)
# ==================================

class CelebAEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(CelebAEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 2048, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(256 * 4 * 4, 2 * latent_dim) # Output mu and log_sigma

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu, log_sigma_unactivated = torch.chunk(x, 2, dim=1)
        sigma = F.softplus(log_sigma_unactivated) # Ensure sigma > 0
        return mu, sigma

class CelebADecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(CelebADecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 2048 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 2048, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x)) # Output in [0, 1] range
        return x

class CelebAVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
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

# --- VAE Loss Function (Not used for training here, but defined for completeness) ---
def vae_loss_function(recon_x, x, mu, sigma):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2))
    return BCE + KL

class ActorNetwork(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, attr_dim=ATTR_DIM):
        super(ActorNetwork, self).__init__()
        self.z_dim = z_dim
        self.use_attr = attr_dim is not None
        input_dim = z_dim
        if self.use_attr:
            self.attr_fc = nn.Linear(attr_dim, 2048)
            input_dim += 2048

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc_out = nn.Linear(2048, 2 * z_dim) # delta_z and gates

    def forward(self, z, y=None):
        z_orig = z
        if self.use_attr and y is not None:
            y_emb = F.relu(self.attr_fc(y))
            x = torch.cat([z, y_emb], dim=1)
        else:
            x = z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)
        delta_z, gate_logits = torch.chunk(x, 2, dim=1)
        gates = torch.sigmoid(gate_logits)
        z_transformed = (1 - gates) * z_orig + gates * delta_z
        return z_transformed


class CriticNetwork(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, attr_dim=ATTR_DIM):
        super(CriticNetwork, self).__init__()
        self.z_dim = z_dim
        self.use_attr = attr_dim is not None
        input_dim = z_dim
        if self.use_attr:
            self.attr_fc = nn.Linear(attr_dim, 2048)
            input_dim += 2048

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc_out = nn.Linear(2048, 1) # Single output score

    def forward(self, z, y=None):
        if self.use_attr and y is not None:
            y_emb = F.relu(self.attr_fc(y))
            x = torch.cat([z, y_emb], dim=1)
        else:
            x = z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)
        output = torch.sigmoid(x) # Output score in [0, 1]
        return output

# ==================================
# 3. Dataset Definition & Loading
# ==================================

# --- Attribute Columns ---
COLUMNS_TO_KEEP = [
    'image_id', 'Blond_Hair', 'Black_Hair', 'Brown_Hair', 'Bald',
    'Eyeglasses', 'No_Beard', 'Wearing_Hat', 'Smiling', 'Male', 'Young'
]
assert len(COLUMNS_TO_KEEP) - 1 == ATTR_DIM, "ATTR_DIM mismatch with COLUMNS_TO_KEEP"

# --- Image Transformations ---
# Transformation for loading images within the dataset
image_transform_load = transforms.Compose([
    # Crop to focus on face, resize to 64x64
    transforms.Lambda(lambda img: transforms.functional.resized_crop(
        img, top=40, left=15, height=148, width=148, size=(64, 64),
        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
    )),
    # Normalize to [0, 1]
    transforms.Lambda(lambda img: img.float() / 255.0),
])

# Transformation specifically for displaying images later
image_transform_display = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.resized_crop(
        img, top=40, left=15, height=148, width=148, size=(64, 64),
        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
    )),
    transforms.Lambda(lambda img: img.float() / 255.0),
])


class CelebALatentDataset(Dataset):
    """
    Custom Dataset that loads images, encodes them using a pre-trained VAE encoder,
    and returns batches of latent vectors paired with attributes.
    NOTE: As requested, the core logic remains the same (encoding within __getitem__).
          This can be slow with num_workers > 0 due to GPU context switching.
          Setting num_workers=0 is recommended here.
    """
    def __init__(self, dataframe, base_path, columns_to_keep, transform, encoder, device, samples_per_image):
        self.dataframe = dataframe
        self.base_path = base_path
        self.columns_to_keep = columns_to_keep
        self.transform = transform
        self.encoder = encoder.to(device).eval() # Ensure encoder is on correct device and in eval mode
        self.device = device
        self.samples_per_image = samples_per_image
        self.sum_sigma = torch.zeros(1, LATENT_DIM, device=device) # For calculating sigma_bar
        self.num_processed = 0 # For calculating sigma_bar

        logging.info(f"CelebALatentDataset initialized with {len(self.dataframe)} images.")
        logging.info(f"Generating {self.samples_per_image} latent samples per image.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.base_path, image_id)

        try:
            img = tv.io.read_image(image_path)
            # Basic check, VAE expects 64x64 input after transform
            if img.shape[1] < 148 or img.shape[2] < 148: # Check original size before crop
                 logging.warning(f"Skipping image {image_path} due to small size {img.shape}")
                 return None # Skip this item

            # Apply transformations (crop, resize, normalize)
            img = self.transform(img)

        except Exception as e:
            logging.error(f"Error loading or transforming image {image_path}: {e}")
            return None # Skip this item

        # Extract attributes (excluding 'image_id')
        attributes = torch.tensor(row[self.columns_to_keep[1:]].values.astype(float), dtype=torch.float32).to(self.device)

        # Move image to device and add batch dimension
        img = img.to(self.device).unsqueeze(0)  # Shape: [1, C, H, W]

        with torch.no_grad():
            # Compute latent parameters (mu, sigma)
            mu, sigma = self.encoder(img)  # Expected shape: [1, latent_dim]

            # Accumulate sigma for calculating the average later
            self.sum_sigma += sigma
            self.num_processed += 1

            # Generate multiple latent samples via reparameterization trick
            eps = torch.randn((self.samples_per_image, LATENT_DIM), device=self.device)
            # mu and sigma have shape [1, latent_dim], eps has [samples, latent_dim]
            # Broadcasting applies: mu + sigma * eps
            z_samples = mu + sigma * eps # Shape: [samples, latent_dim]

            # Expand attributes to match the number of samples
            attrs_expanded = attributes.unsqueeze(0).expand(self.samples_per_image, -1)  # [samples, attr_dim]

            # Concatenate latent vectors and attributes
            # final_vectors shape: [samples, latent_dim + attr_dim]
            final_vectors = torch.cat((z_samples, attrs_expanded), dim=1)

        return final_vectors

    def get_average_sigma(self):
        if self.num_processed == 0:
            logging.warning("Cannot calculate average sigma, no images processed.")
            return torch.ones(1, LATENT_DIM, device=self.device) # Return default
        avg_sigma = self.sum_sigma / self.num_processed
        logging.info(f"Calculated average sigma (sigma_bar) from {self.num_processed} images.")
        # Detach from graph and potentially move to CPU if needed elsewhere, but keep on device for training
        return avg_sigma.detach()


def load_data(attr_csv_path, image_dir, columns_to_keep):
    """Loads the attribute CSV."""
    try:
        df = pd.read_csv(attr_csv_path)
        df = df.replace(-1, 0) # Convert attribute values from {-1, 1} to {0, 1}
        df = df[columns_to_keep]
        logging.info(f"Loaded attributes CSV from {attr_csv_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Attribute CSV not found at {attr_csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading attributes CSV: {e}")
        raise

def create_latent_dataloader(dataframe, base_path, columns_to_keep, transform, encoder, device, batch_size, samples_per_image, num_workers=0):
    """Creates the Dataset and DataLoader for latent vectors."""
    dataset = CelebALatentDataset(dataframe, base_path, columns_to_keep, transform, encoder, device, samples_per_image)
    
    # Custom collate function to handle None values returned by __getitem__
    def collate_fn(batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid
        # Stack the valid items
        return torch.cat(batch, dim=0) # Cat samples from different images

    # IMPORTANT: Use num_workers=0 if encoding is done on GPU inside __getitem__
    if num_workers > 0 and device.type == 'cuda':
         logging.warning("Using num_workers > 0 with GPU encoding in __getitem__ might be slow or cause errors. Setting num_workers=0 is recommended.")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False, # Shuffle=False helps calculate sigma_bar consistently
                            num_workers=num_workers,
                            collate_fn=collate_fn, # Use custom collate function
                            drop_last=False) # Keep all data
    logging.info(f"Created DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    return dataloader, dataset # Return dataset to calculate sigma_bar


# ==================================
# 4. Training Logic (Actor/Critic)
# ==================================

# --- Loss Functions for Actor/Critic ---
criterion_gan = nn.BCELoss() # For GAN realism loss

def critic_loss_fn(critic_net, actor_net, z_real, z_prior, attributes):
    """Calculates the critic loss."""
    # Detach actor output to prevent gradients flowing back during critic update
    with torch.no_grad():
         z_fake = actor_net(z_prior, attributes).detach()

    # Critic score for real latent vectors (target = 1)
    d_real = critic_net(z_real, attributes)
    target_real = torch.ones_like(d_real, device=d_real.device)
    loss_real = criterion_gan(d_real, target_real)

    # Critic score for fake latent vectors (generated, target = 0)
    d_fake = critic_net(z_fake, attributes)
    target_fake = torch.zeros_like(d_fake, device=d_fake.device)
    loss_fake = criterion_gan(d_fake, target_fake)

    # Critic score for prior latent vectors (sampled, target = 0)
    d_prior = critic_net(z_prior, attributes)
    loss_prior = criterion_gan(d_prior, target_fake) # Also treat prior as "not real"

    total_loss = loss_real + loss_fake + loss_prior
    return total_loss


def generator_loss_fn(critic_net, actor_net, z_prior, attributes, sigma_bar, lambda_dist):
    """Calculates the generator (actor) loss."""
    # Generate fake latent vectors
    z_fake = actor_net(z_prior, attributes)

    # Realism loss: Encourage critic to believe generated codes are real (target = 1)
    d_fake = critic_net(z_fake, attributes)
    target_real = torch.ones_like(d_fake, device=d_fake.device)
    realism_loss = criterion_gan(d_fake, target_real)

    # Distance penalty (regularization)
    # Ensure sigma_bar has the correct shape and avoid division by zero
    sigma_bar_sq = sigma_bar.pow(2).clamp(min=1e-6) # Add small epsilon for stability
    # Calculate element-wise loss, then average
    distance_loss = (1.0 / sigma_bar_sq) * torch.log(1 + (z_fake - z_prior).pow(2))
    distance_loss = distance_loss.mean() # Average over batch and latent dim

    total_loss = realism_loss + lambda_dist * distance_loss
    return total_loss


def train_actor_critic(dataloader, sigma_bar, epochs, lr, lambda_dist, device, model_save_dir):
    """Trains the Actor and Critic networks."""
    logging.info("Starting Actor/Critic training...")
    start_time = time.time()

    actor = ActorNetwork(z_dim=LATENT_DIM, attr_dim=ATTR_DIM).to(device)
    critic = CriticNetwork(z_dim=LATENT_DIM, attr_dim=ATTR_DIM).to(device)

    optimizer_actor = Adam(actor.parameters(), lr=lr)
    optimizer_critic = Adam(critic.parameters(), lr=lr)

    # Ensure sigma_bar is on the correct device and has the right shape for broadcasting
    sigma_bar = sigma_bar.to(device).view(1, LATENT_DIM) # Shape [1, latent_dim]

    for epoch in range(epochs):
        actor.train()
        critic.train()
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        batches_processed = 0

        for i, batch_data in enumerate(dataloader):
             # Skip if batch is None (due to collate_fn handling errors)
            if batch_data is None:
                 logging.warning(f"Skipping empty batch {i} in epoch {epoch+1}")
                 continue

            # batch_data has shape [N * samples_per_image, latent_dim + attr_dim]
            batch_data = batch_data.to(device)
            current_batch_size = batch_data.size(0) # Actual batch size after collation/skipping

            # Split batch into real latent vectors (z_real) and attributes (y)
            z_real = batch_data[:, :LATENT_DIM]
            attributes = batch_data[:, LATENT_DIM:]

            # Sample from prior distribution p(z) - standard normal
            z_prior = torch.randn(current_batch_size, LATENT_DIM, device=device)

            # ---------------- Train Critic ----------------
            optimizer_critic.zero_grad()
            loss_critic = critic_loss_fn(critic, actor, z_real, z_prior, attributes)
            loss_critic.backward()
            optimizer_critic.step()

            # ---------------- Train Actor -----------------
            optimizer_actor.zero_grad()
            loss_actor = generator_loss_fn(critic, actor, z_prior, attributes, sigma_bar, lambda_dist)
            loss_actor.backward()
            optimizer_actor.step()

            epoch_actor_loss += loss_actor.item()
            epoch_critic_loss += loss_critic.item()
            batches_processed += 1

            if (i + 1) % 100 == 0: # Log progress every 100 batches
                 logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], "
                              f"Actor Loss: {loss_actor.item():.4f}, Critic Loss: {loss_critic.item():.4f}")


        # --- End of Epoch ---
        avg_actor_loss = epoch_actor_loss / batches_processed if batches_processed > 0 else 0
        avg_critic_loss = epoch_critic_loss / batches_processed if batches_processed > 0 else 0
        logging.info(f"--- Epoch {epoch+1}/{epochs} Completed ---")
        logging.info(f"Average Actor Loss: {avg_actor_loss:.4f}")
        logging.info(f"Average Critic Loss: {avg_critic_loss:.4f}")

        # Save model checkpoints
        actor_save_path = os.path.join(model_save_dir, f'actor_model_epoch_{epoch+1}.pth')
        critic_save_path = os.path.join(model_save_dir, f'critic_model_epoch_{epoch+1}.pth')
        torch.save(actor.state_dict(), actor_save_path)
        torch.save(critic.state_dict(), critic_save_path)
        # logging.info(f"Saved models to {actor_save_path} and {critic_save_path}")

    # --- End of Training ---
    end_time = time.time()
    training_duration = end_time - start_time
    logging.info(f"Actor/Critic training finished in {training_duration:.2f} seconds.")

    # Log the final training time
    global GENERATOR_TRAIN_TIME
    GENERATOR_TRAIN_TIME = training_duration

    return actor, critic # Return trained models


# ==================================
# 5. Evaluation and Visualization
# ==================================

def generate_and_save_comparison_images(vae_decoder, actor_net, image_paths, attributes_df, columns_to_keep, device, output_dir, filename="comparison_images.png"):
    """Generates and saves comparison images: Original, Reconstructed, Attribute-Modified."""
    logging.info(f"Generating comparison images for {len(image_paths)} samples...")

    vae_decoder.to(device).eval()
    actor_net.to(device).eval()

    n_images = len(image_paths)
    if n_images == 0:
        logging.warning("No images provided for comparison.")
        return

    # Define the target attribute vectors (one-hot for each attribute)
    attribute_vectors = torch.eye(ATTR_DIM, dtype=torch.float32).to(device)
    attribute_names = columns_to_keep[1:] # Get names matching the vectors

    n_attributes_to_show = len(attribute_vectors)
    n_cols = 2 + n_attributes_to_show # Original, Reconstructed, + N attributes
    n_rows = n_images

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: # Should not happen with current setup but good practice
         axes = np.expand_dims(axes, axis=1)


    loaded_images = []
    latent_vectors = []

    # --- Load and Encode Original Images ---
    logging.info("Loading and encoding original images...")
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = tv.io.read_image(img_path)
                img_transformed = image_transform_display(img) # Use display transform
                loaded_images.append(img_transformed) # Keep CPU tensor for plotting

                # Encode the image to get its latent representation (using VAE encoder implicitly via VAE forward then taking mu)
                # We need the VAE model here to get mu, sigma first.
                # For simplicity, let's assume the VAE model is available globally or passed in.
                # Re-using the loaded VAE instance
                mu, sigma = vae_model.encoder(img_transformed.unsqueeze(0).to(device))
                # Use the mean for reconstruction and modification base
                z = mu # Or use reparameterize: vae_model.reparameterize(mu, sigma)
                latent_vectors.append(z)

            except Exception as e:
                logging.error(f"Failed to load/encode image {img_path}: {e}")
                # Fill plots with black squares or skip row if needed
                for j in range(n_cols):
                    axes[i, j].imshow(torch.zeros_like(img_transformed).permute(1, 2, 0))
                    axes[i, j].set_title("Error")
                    axes[i, j].axis("off")
                continue # Skip to next image

    # --- Generate and Plot ---
    logging.info("Generating reconstructed and modified images...")
    with torch.no_grad():
        for i in range(n_rows):
            if i >= len(loaded_images): continue # Skip if loading failed

            original_img_cpu = loaded_images[i]
            latent_vec = latent_vectors[i] # Shape [1, latent_dim]

            # Column 0: Original Image
            axes[i, 0].imshow(original_img_cpu.permute(1, 2, 0))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            # Column 1: Reconstructed Image
            recon_img = vae_decoder(latent_vec).squeeze(0).cpu()
            axes[i, 1].imshow(recon_img.permute(1, 2, 0))
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")

            # Columns 2+: Attribute Modifications
            for j, (attr_vec, attr_name) in enumerate(zip(attribute_vectors, attribute_names)):
                attr_vec = attr_vec.unsqueeze(0) # Add batch dimension [1, attr_dim]

                # Modify latent vector using Actor network
                modified_latent = actor_net(latent_vec, attr_vec)

                # Decode modified latent vector
                modified_img = vae_decoder(modified_latent).squeeze(0).cpu()

                axes[i, j + 2].imshow(modified_img.permute(1, 2, 0))
                axes[i, j + 2].set_title(f"Add: {attr_name}")
                axes[i, j + 2].axis("off")

    plt.tight_layout(pad=0.5) # Add a little padding
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    logging.info(f"Comparison image saved to {save_path}")
    # plt.show() # Optionally display the plot interactively
    plt.close(fig) # Close the figure to free memory


# ==================================
# 6. Main Execution Block
# ==================================

if __name__ == "__main__":
    logging.info(" --- Starting Main Execution --- ")

    # --- Load VAE Model ---
    logging.info(f"Loading pre-trained VAE model from {VAE_MODEL_PATH}...")
    try:
        vae_model = CelebAVAE(latent_dim=LATENT_DIM)
        vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        vae_model.to(DEVICE).eval() # Set to evaluation mode
        encoder = vae_model.encoder
        decoder = vae_model.decoder
        logging.info("Pre-trained VAE model loaded successfully.")
        logging.info(f"VAE Training Time: Not available (loaded pre-trained model)") # Log VAE time info
    except FileNotFoundError:
        logging.error(f"VAE model file not found at {VAE_MODEL_PATH}. Exiting.")
        exit()
    except Exception as e:
        logging.error(f"Error loading VAE model: {e}. Exiting.")
        exit()

    # --- Prepare Latent Dataset ---
    logging.info("Preparing latent vector dataset...")
    attribute_data = load_data(ATTR_CSV_PATH, IMAGE_DIR, COLUMNS_TO_KEEP)
    latent_dataloader, latent_dataset = create_latent_dataloader(
        dataframe=attribute_data,
        base_path=IMAGE_DIR,
        columns_to_keep=COLUMNS_TO_KEEP,
        transform=image_transform_load,
        encoder=encoder,
        device=DEVICE,
        batch_size=DS_BATCH_SIZE,
        samples_per_image=DS_SAMPLES_PER_IMAGE,
        num_workers=0 # Recommended for GPU encoding in __getitem__
    )

    # --- Calculate Average Sigma (sigma_bar) ---
    # This requires iterating through the dataloader once if not pre-calculated
    logging.info("Calculating average sigma (sigma_bar) across dataset...")
    # Reset accumulator before iteration (in case dataset was iterated before)
    latent_dataset.sum_sigma.zero_()
    latent_dataset.num_processed = 0
    # Iterate through dataloader to populate sigma calculation in dataset.__getitem__
    # This loop doesn't do training, just triggers __getitem__ calls
    temp_count = 0
    sigma_calc_start = time.time()
    for _ in latent_dataloader:
        temp_count += 1
        if temp_count % 500 == 0:
             logging.info(f"  Processed {temp_count * DS_BATCH_SIZE} images for sigma calculation...")
        pass # Just need to iterate
    sigma_bar = latent_dataset.get_average_sigma()
    sigma_calc_end = time.time()
    logging.info(f"Sigma_bar calculation took {sigma_calc_end - sigma_calc_start:.2f} seconds.")
    # logging.info(f"Calculated sigma_bar (first few values): {sigma_bar.squeeze()[:5].cpu().numpy()}") # Log first few values

    # --- Train Actor and Critic ---
    # Reset dataloader shuffle=True for actual training if desired,
    # but False is okay too for this type of GAN training on latent space.
    # Using shuffle=False as defined earlier.
    trained_actor, trained_critic = train_actor_critic(
        dataloader=latent_dataloader,
        sigma_bar=sigma_bar,
        epochs=AC_EPOCHS,
        lr=AC_LEARNING_RATE,
        lambda_dist=AC_LAMBDA_DIST,
        device=DEVICE,
        model_save_dir=MODEL_SAVE_DIR
    )

    # --- Generate Comparison Images ---
    # Select sample images (Using the same indices as the original script)
    sample_indices = [6577, 1129, 6539, 12993, 19284, 7499, 5899, 8547, 13342, 728]
    logging.info(f"Selecting {len(sample_indices)} sample images for final visualization: {sample_indices}")
    sample_image_data = attribute_data.iloc[sample_indices]
    sample_image_ids = sample_image_data['image_id'].tolist()
    sample_image_paths = [os.path.join(IMAGE_DIR, img_id) for img_id in sample_image_ids]

    # Load the *best* or *final* actor model for generation
    # Example: Load the last saved actor model
    final_actor_path = os.path.join(MODEL_SAVE_DIR, f'actor_model_epoch_{AC_EPOCHS}.pth')
    if os.path.exists(final_actor_path):
        logging.info(f"Loading actor model from {final_actor_path} for visualization.")
        trained_actor.load_state_dict(torch.load(final_actor_path, map_location=DEVICE))
    else:
         logging.warning(f"Final actor model {final_actor_path} not found. Using model state from end of training.")


    generate_and_save_comparison_images(
        vae_decoder=decoder,
        actor_net=trained_actor,
        image_paths=sample_image_paths,
        attributes_df=attribute_data, # Pass the full df if needed later, though not used currently
        columns_to_keep=COLUMNS_TO_KEEP,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        filename="final_comparison_images.png"
    )

    # --- Final Logging ---
    logging.info("=" * 40)
    logging.info("Script Execution Summary")
    logging.info(f"VAE Training Time: {VAE_TRAIN_TIME if VAE_TRAIN_TIME else 'N/A (Pre-trained)'}")
    logging.info(f"Generator (Actor/Critic) Training Time: {GENERATOR_TRAIN_TIME:.2f} seconds" if GENERATOR_TRAIN_TIME else "N/A (Training failed or skipped)")
    logging.info(f"Outputs (logs, models, images) saved in: {OUTPUT_DIR}")
    logging.info("Script Execution Finished")
    logging.info("=" * 40)