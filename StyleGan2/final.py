import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import open_clip
import dnnlib
import legacy
import logging

# --- Configuration ---
OUTPUT_BASE_DIR = "final_output"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LOG_FILE = os.path.join(OUTPUT_BASE_DIR, "log.txt")
NETWORK_PKL = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl"
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_TO_GENERATE = 50000 # Keep moderate for memory/time
STYLEGAN_BATCH_SIZE = 32       # Batch size for StyleGAN generation
VIT_BATCH_SIZE = 64            # Batch size for ViT feature extraction (adjust based on GPU memory)
VIT_MODEL_NAME = 'ViT-B-32'
VIT_PRETRAINED_DATASET = 'laion2b_s34b_b79k'
TARGET_IMG_SIZE = (244, 244)   # Target size for ViT input

# Clustering and Latent GAN params
N_CLUSTERS = 10
CLUSTER_TO_TRAIN_ON = 8 # Example: Train on cluster 8
LATENT_GAN_EPOCHS = 1000 # Increased epochs for potentially better results
LATENT_GAN_BATCH_SIZE = 64
LATENT_GAN_LR = 0.0002
LATENT_DIM = 512
# --- End Configuration ---

# --- Setup Output Directories and Logging ---
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # Overwrite log file each run
        logging.StreamHandler()                  # Also print logs to console
    ]
)
logging.info("--- Script Execution Started ---")
logging.info(f"Using device: {DEVICE}")
logging.info(f"Output directory: {OUTPUT_BASE_DIR}")
logging.info(f"Image directory: {IMG_DIR}")
logging.info(f"Log file: {LOG_FILE}")
# --- End Setup ---


# --- Helper Functions ---
def load_stylegan(network_pkl, device):
    """Loads the StyleGAN2 generator."""
    logging.info(f"Loading StyleGAN2 generator from {network_pkl}...")
    try:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        G.eval()
        for param in G.parameters():
            param.requires_grad = False
        logging.info("StyleGAN2 generator loaded successfully.")
        return G
    except Exception as e:
        logging.error(f"Failed to load StyleGAN2: {e}", exc_info=True)
        raise

def generate_stylegan_data(G, num_images, batch_size, target_size, device):
    """Generates images and corresponding latent vectors using StyleGAN."""
    logging.info(f"Generating {num_images} images with StyleGAN...")
    start_time = time.time()
    latent_vectors_z = []
    all_images_tensor = [] # Store tensors directly for efficiency if memory allows
    num_batches = (num_images + batch_size - 1) // batch_size # Handle non-divisible cases

    G.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_images - i * batch_size)
            if current_batch_size <= 0:
                break

            z = torch.randn(current_batch_size, G.z_dim, device=device)
            w = G.mapping(z, None)
            generated_images = G.synthesis(w, noise_mode='const')

            # Resize images
            resized_images = F.interpolate(generated_images, size=target_size, mode='bilinear', align_corners=False)

            # Normalize images to [0, 1] range expected by ViT preprocess/CLIP
            # CLIP models usually expect normalization based on ImageNet stats,
            # but let's keep the [0, 1] for now as in the original code's feature extraction prep.
            # Revisit if ViT performance is poor.
            normalized_images = (resized_images.clamp(-1, 1) + 1) / 2

            latent_vectors_z.append(z.cpu()) # Store latents on CPU to save GPU RAM
            all_images_tensor.append(normalized_images.cpu()) # Store images on CPU

            if (i + 1) % (max(1, num_batches // 10)) == 0: # Log progress roughly 10 times
                 logging.info(f"  Generated {(i + 1) * batch_size}/{num_images} images...")

    # Concatenate lists of tensors into single tensors
    latent_vectors_z = torch.cat(latent_vectors_z, dim=0)
    all_images_tensor = torch.cat(all_images_tensor, dim=0)

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Finished generating {latent_vectors_z.shape[0]} images and latents.")
    logging.info(f"Image generation time: {duration:.2f} seconds")

    # Log time to file
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n--- Timing Metrics ---\n")
        f.write(f"Image Generation Time: {duration:.2f} seconds\n")

    return latent_vectors_z, all_images_tensor # Return tensors

def load_vit_model(model_name, pretrained_dataset, device):
    """Loads the OpenCLIP ViT model and preprocessing."""
    logging.info(f"Loading OpenCLIP model: {model_name} ({pretrained_dataset})...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_dataset,
            device=device # Load directly to the target device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        logging.info("OpenCLIP model loaded successfully.")
        # Note: The preprocess function returned here often includes normalization
        # specific to the model's training (like ImageNet stats).
        # Our image generation creates [0, 1] images. For best ViT results,
        # we *should* apply the specific normalization from 'preprocess'.
        # However, the original code didn't, so we'll stick to that for now,
        # but be aware this is a potential point for improvement.
        return model, preprocess, tokenizer
    except Exception as e:
        logging.error(f"Failed to load OpenCLIP model: {e}", exc_info=True)
        raise

def extract_vit_features(model, images_tensor, batch_size, device):
    """Extracts ViT features from a tensor of images."""
    logging.info(f"Extracting ViT features from {images_tensor.shape[0]} images...")
    start_time = time.time()
    num_images = images_tensor.shape[0]
    num_batches = (num_images + batch_size - 1) // batch_size
    all_features = []

    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            batch_images = images_tensor[start_idx:end_idx].to(device) # Move batch to GPU

            # Apply model-specific normalization if needed (using preprocess)
            # For now, assuming model works okay with [0, 1] images
            # Example if using preprocess:
            # batch_images = preprocess(batch_images) # This usually handles resizing and normalization

            # Use autocast for potential speedup with mixed precision
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                 features = model.encode_image(batch_images)
                 features /= features.norm(dim=-1, keepdim=True) # Normalize features

            all_features.append(features.cpu()) # Move features to CPU

            if (i + 1) % (max(1, num_batches // 10)) == 0: # Log progress
                 logging.info(f"  Processed {end_idx}/{num_images} images for feature extraction...")

    image_features = torch.cat(all_features, dim=0).numpy() # Combine features into a NumPy array

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Finished extracting {image_features.shape[0]} features.")
    logging.info(f"ViT feature extraction time: {duration:.2f} seconds")

    with open(LOG_FILE, 'a') as f:
        f.write(f"ViT Feature Extraction Time: {duration:.2f} seconds\n")

    return image_features


def cluster_features(features, n_clusters, img_save_path):
    """Performs dimensionality reduction, clustering, and saves visualization."""
    logging.info("Performing dimensionality reduction and clustering...")
    start_time = time.time()

    # Optional: PCA for initial dimensionality reduction (can speed up t-SNE)
    if features.shape[1] > 50:
        logging.info("Applying PCA to reduce dimensions to 50...")
        pca = PCA(n_components=50, random_state=42)
        features_reduced = pca.fit_transform(features)
        logging.info("PCA complete.")
    else:
        features_reduced = features

    # t-SNE for 2D visualization
    logging.info("Applying t-SNE for 2D visualization (perplexity=30, n_iter=1000)...")
    # Consider adjusting perplexity based on dataset size (e.g., 30-50 is common)
    # n_iter can be increased for potentially better convergence
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_reduced)
    logging.info("t-SNE complete.")

    # KMeans Clustering on original or PCA-reduced features (usually better than t-SNE features)
    logging.info(f"Applying KMeans clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init='auto' in newer sklearn
    cluster_labels = kmeans.fit_predict(features) # Cluster on original features
    logging.info("KMeans clustering complete.")

    # Plotting t-SNE results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=cluster_labels, cmap='viridis', alpha=0.7, s=10 # Smaller points for large datasets
    )
    plt.title(f"t-SNE Visualization of {n_clusters} Clusters (based on ViT features)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    plot_filename = os.path.join(img_save_path, "tsne_clusters.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    logging.info(f"t-SNE cluster visualization saved to {plot_filename}")
    plt.close() # Close the figure to free memory

    end_time = time.time()
    logging.info(f"Clustering and visualization time: {end_time - start_time:.2f} seconds")

    return cluster_labels

def group_latents_by_cluster(latent_vectors_z, cluster_labels, n_clusters):
    """Groups latent vectors according to their assigned cluster label."""
    logging.info("Grouping latent vectors by cluster label...")
    latents_by_cluster = {i: [] for i in range(n_clusters)}
    for latent, label in zip(latent_vectors_z, cluster_labels):
        latents_by_cluster[label].append(latent) # Keep as tensors

    # Convert lists of tensors to single tensors per cluster
    for label in latents_by_cluster:
        if latents_by_cluster[label]:
            latents_by_cluster[label] = torch.stack(latents_by_cluster[label])
            logging.info(f"  Cluster {label}: {latents_by_cluster[label].shape[0]} vectors")
        else:
            logging.warning(f"  Cluster {label}: 0 vectors found!")
            latents_by_cluster[label] = torch.empty((0, latent_vectors_z.shape[1])) # Empty tensor

    return latents_by_cluster


def save_cluster_sample_images(G, latents_by_cluster, num_samples_per_cluster, img_save_path, device):
    """Generates and saves sample images for each cluster."""
    logging.info("Generating and saving sample images for each cluster...")
    G.eval()
    num_to_show = min(num_samples_per_cluster, 5) # Limit samples shown inline for brevity

    for cluster_label, latents_tensor in latents_by_cluster.items():
        if latents_tensor.shape[0] == 0:
            logging.warning(f"Skipping cluster {cluster_label} - no latents.")
            continue

        logging.info(f"  Generating samples for cluster {cluster_label}...")
        num_available = latents_tensor.shape[0]
        num_generate = min(num_samples_per_cluster, num_available)

        # Select random latent vectors from the cluster
        indices = torch.randperm(num_available)[:num_generate]
        selected_latents = latents_tensor[indices].to(device) # Move selected to GPU

        generated_images_list = []
        with torch.no_grad():
             # Generate images one by one or in small batches if num_generate is large
             # Generating one-by-one is simpler here
             for i in range(num_generate):
                z_sample = selected_latents[i:i+1] # Keep batch dim
                w_sample = G.mapping(z_sample, None)
                img_sample = G.synthesis(w_sample, noise_mode='const')
                # Normalize to [0, 1] for saving with save_image
                img_sample = (img_sample.clamp(-1, 1) + 1) / 2
                generated_images_list.append(img_sample.cpu()) # Move back to CPU

        # Save images as a grid
        if generated_images_list:
            grid = vutils.make_grid(torch.cat(generated_images_list, dim=0), nrow=int(np.sqrt(num_generate)) or 5, padding=2, normalize=False) # normalize=False since already [0,1]
            grid_filename = os.path.join(img_save_path, f"cluster_{cluster_label}_samples.png")
            vutils.save_image(grid, grid_filename)
            logging.info(f"    Saved sample grid to {grid_filename}")

            # Optional: Show a few samples directly if running interactively (might need plt.show())
            # if cluster_label < 3: # Show first 3 clusters
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(grid.permute(1, 2, 0))
            #     plt.title(f"Cluster {cluster_label} Samples")
            #     plt.axis('off')
            #     plt.show()


# --- Latent GAN Definition ---
class SimpleLatentGenerator(nn.Module):
    def __init__(self, input_dim=LATENT_DIM, output_dim=LATENT_DIM, hidden_dim=1024):
        super(SimpleLatentGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True), # Use LeakyReLU often better in GANs
            nn.BatchNorm1d(hidden_dim),      # Add BatchNorm
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            # No Tanh needed if original latents are standard normal
            # If StyleGAN Z is not standard normal, adjust output activation
        )

    def forward(self, x):
        return self.model(x)

# --- Latent GAN Training ---
def train_latent_gan(target_latents_tensor, num_epochs, batch_size, learning_rate, device):
    """Trains the simple generator on the target latent vectors."""
    if target_latents_tensor.shape[0] < batch_size:
        logging.warning(f"Target cluster has only {target_latents_tensor.shape[0]} vectors. Reducing batch size for training.")
        batch_size = target_latents_tensor.shape[0]

    if batch_size == 0:
        logging.error("Cannot train Latent GAN: Target cluster has 0 vectors.")
        return None

    logging.info(f"Training Latent GAN for {num_epochs} epochs...")
    start_time = time.time()

    latent_gen = SimpleLatentGenerator(input_dim=LATENT_DIM, output_dim=LATENT_DIM).to(device)
    latent_gen.train() # Set to training mode

    optimizer = optim.Adam(latent_gen.parameters(), lr=learning_rate, betas=(0.5, 0.999)) # Betas common for GANs
    criterion = nn.MSELoss() # Simple MSE loss to match target distribution moments

    target_latents_tensor = target_latents_tensor.to(device) # Ensure targets are on device
    dataset = torch.utils.data.TensorDataset(target_latents_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches_epoch = 0
        for i, (real_latents,) in enumerate(dataloader):
            # real_latents is already on device if target_latents_tensor was moved
            current_batch_size = real_latents.size(0)

            # Generate fake latents
            noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
            fake_latents = latent_gen(noise)

            # Calculate loss - how close are fake latents to real ones?
            loss = criterion(fake_latents, real_latents) # Match generated to real samples

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches_epoch += 1

        avg_epoch_loss = epoch_loss / num_batches_epoch
        losses.append(avg_epoch_loss)

        if (epoch + 1) % max(1, num_epochs // 10) == 0: # Log progress
            logging.info(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Finished training Latent GAN.")
    logging.info(f"Latent GAN training time: {duration:.2f} seconds")

    with open(LOG_FILE, 'a') as f:
        f.write(f"Latent GAN Training Time: {duration:.2f} seconds\n")
        f.write(f"Final Latent GAN Loss: {losses[-1]:.4f}\n")

    # Optional: Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Latent GAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    loss_plot_filename = os.path.join(OUTPUT_BASE_DIR, "latent_gan_training_loss.png")
    plt.savefig(loss_plot_filename)
    logging.info(f"Latent GAN training loss plot saved to {loss_plot_filename}")
    plt.close()

    latent_gen.eval() # Set back to eval mode
    return latent_gen

# --- Latent GAN Evaluation ---
def evaluate_latent_gan(latent_generator, stylegan_G, num_images, img_save_path, device):
    """Generates images using the trained latent GAN and StyleGAN."""
    if latent_generator is None:
        logging.error("Latent generator is None, cannot evaluate.")
        return

    logging.info(f"Generating {num_images} images using the trained Latent GAN...")
    latent_generator.eval()
    stylegan_G.eval()

    generated_images_list = []
    with torch.no_grad():
        for _ in range(num_images):
            # Generate a latent vector using the trained generator
            noise = torch.randn(1, LATENT_DIM, device=device)
            generated_z = latent_generator(noise) # This z should mimic the target cluster's latents

            # Generate image using StyleGAN
            w = stylegan_G.mapping(generated_z, None)
            img = stylegan_G.synthesis(w, noise_mode='const')
            # Normalize to [0, 1] for saving
            img = (img.clamp(-1, 1) + 1) / 2
            generated_images_list.append(img.cpu())

    # Save images as a grid
    if generated_images_list:
        grid = vutils.make_grid(torch.cat(generated_images_list, dim=0), nrow=int(np.sqrt(num_images)) or 5, padding=2, normalize=False)
        grid_filename = os.path.join(img_save_path, "final_latent_gan_results.png")
        vutils.save_image(grid, grid_filename)
        logging.info(f"Saved final results grid ({num_images} images) to {grid_filename}")


# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()

    # 1. Load StyleGAN
    stylegan_G = load_stylegan(NETWORK_PKL, DEVICE)

    # 2. Generate Initial Data (Images + Latents)
    # Consider reducing NUM_IMAGES_TO_GENERATE if memory is an issue
    latent_vectors_z, images_tensor = generate_stylegan_data(
        stylegan_G, NUM_IMAGES_TO_GENERATE, STYLEGAN_BATCH_SIZE, TARGET_IMG_SIZE, DEVICE
    )
    # latent_vectors_z shape: (N, 512) - on CPU
    # images_tensor shape: (N, 3, 244, 244) - on CPU, range [0, 1]

    # 3. Load ViT Model
    vit_model, vit_preprocess, _ = load_vit_model(VIT_MODEL_NAME, VIT_PRETRAINED_DATASET, DEVICE)

    # 4. Extract ViT Features
    # Pass images_tensor directly
    image_features_np = extract_vit_features(vit_model, images_tensor, VIT_BATCH_SIZE, DEVICE)
    # image_features_np shape: (N, feature_dim) - on CPU (numpy)

    # Free up memory from images tensor if no longer needed (important for large N)
    del images_tensor
    import gc
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    logging.info("Released image tensor memory.")

    # 5. Cluster Features and Visualize
    cluster_labels = cluster_features(image_features_np, N_CLUSTERS, IMG_DIR)
    # cluster_labels shape: (N,) - on CPU (numpy)

    # Free up memory from features if no longer needed (clustering done)
    del image_features_np
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    logging.info("Released feature vector memory.")

    # 6. Group Latent Vectors by Cluster
    # Pass the original Z latents (on CPU) and cluster labels (numpy)
    latents_by_cluster = group_latents_by_cluster(latent_vectors_z, cluster_labels, N_CLUSTERS)
    # latents_by_cluster: dict {label: tensor} - tensors on CPU

    # 7. Save Sample Images for Each Cluster
    save_cluster_sample_images(stylegan_G, latents_by_cluster, num_samples_per_cluster=25, img_save_path=IMG_DIR, device=DEVICE)

    # 8. Train Latent GAN on a Specific Cluster
    if CLUSTER_TO_TRAIN_ON in latents_by_cluster and latents_by_cluster[CLUSTER_TO_TRAIN_ON].shape[0] > 0:
        target_latents = latents_by_cluster[CLUSTER_TO_TRAIN_ON] # Get tensor for the target cluster (on CPU)
        trained_latent_generator = train_latent_gan(
            target_latents,
            num_epochs=LATENT_GAN_EPOCHS,
            batch_size=LATENT_GAN_BATCH_SIZE,
            learning_rate=LATENT_GAN_LR,
            device=DEVICE
        )

        # 9. Evaluate the Trained Latent GAN
        if trained_latent_generator:
             evaluate_latent_gan(trained_latent_generator, stylegan_G, num_images=25, img_save_path=IMG_DIR, device=DEVICE)
        else:
             logging.error("Latent GAN training failed, skipping evaluation.")

    else:
        logging.error(f"Cannot train Latent GAN: Cluster {CLUSTER_TO_TRAIN_ON} not found or has no vectors.")


    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    logging.info(f"--- Script Execution Finished ---")
    logging.info(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    with open(LOG_FILE, 'a') as f:
        f.write(f"\nTotal Execution Time: {total_duration:.2f} seconds\n")