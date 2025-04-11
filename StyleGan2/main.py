# -----------------------imports----------------------
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # For progress bars

# StyleGAN / dnnlib specific imports - ensure dnnlib is accessible
try:
    import dnnlib
    import legacy # Assuming legacy.py is available alongside dnnlib
except ImportError:
    print("Error: dnnlib or legacy.py not found. Make sure StyleGAN2-ADA Pytorch repo is cloned and in PYTHONPATH.")
    exit()

# Feature Extractor (OpenCLIP)
try:
    import open_clip
except ImportError:
    print("Error: open_clip not found. Install with 'pip install open_clip_torch'.")
    exit()

# Clustering and Dimensionality Reduction
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA # Optional but kept for potential use

# -----------------------Configuration----------------------
# --- General ---
DEVICE_ID = "cuda:0" # Choose your desired GPU ID if multiple are available
SEED = 42
OUTPUT_BASE_DIR = "stylegan_clustering_output"
LOG_FILE = os.path.join(OUTPUT_BASE_DIR, "experiment.log")

# --- StyleGAN ---
# NETWORK_PKL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
NETWORK_PKL = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl" # Original Car model
TRUNCATION_PSI = 0.7 # Common value for higher quality images, 1.0 for more diversity

# --- Image Generation ---
NUM_IMAGES_TO_GENERATE = 1000 # Reduced for faster testing, change back to 10000 if needed
GENERATION_BATCH_SIZE = 16 # Adjust based on GPU memory for StyleGAN
SAVE_IMAGES = True # Set to False if you don't need individual image files
GENERATED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_images")

# --- Feature Extraction ---
VIT_MODEL_NAME = 'ViT-B-32'
VIT_PRETRAINED = 'laion2b_s34b_b79k'
FEATURE_EXTRACTION_BATCH_SIZE = 64 # Adjust based on GPU memory for ViT

# --- Clustering ---
NUM_CLUSTERS = 10
USE_PCA_BEFORE_TSNE = False # Set to True to use PCA for dimensionality reduction before t-SNE
PCA_COMPONENTS = 50 # Number of components if PCA is used
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
CLUSTER_VIS_SAMPLES = 5 # Number of sample images to show per cluster

# --- Simple Generator Training (Optional Section from original code) ---
TRAIN_SIMPLE_GENERATOR = False # Set to True to run this part
SIMPLE_GENERATOR_EPOCHS = 5
SIMPLE_GENERATOR_BATCH_SIZE = 64
SIMPLE_GENERATOR_TARGET_CLUSTER = 8 # Which cluster to train the generator on

# -----------------------Setup----------------------
def setup_environment(seed, log_file, base_dir, generated_images_dir):
    """Sets up logging, directories, and seeds."""
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Log to console as well
        ]
    )
    logging.info("--- Experiment Start ---")
    logging.info(f"Output directory: {base_dir}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")

def get_device(device_id):
    """Gets the torch device."""
    if torch.cuda.is_available():
        device = torch.device(device_id)
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

# -----------------------Model Loading----------------------
def load_stylegan(network_pkl, device):
    """Loads the StyleGAN2 generator model."""
    logging.info(f"Loading StyleGAN2 model from: {network_pkl}")
    try:
        with dnnlib.util.open_url(network_pkl) as f:
            # G = legacy.load_network_pkl(f)['G_ema'].to(device) # For TF weights
             # Adjust based on the pickle structure, common for PyTorch versions:
            data = legacy.load_network_pkl(f)
            G = data.get('G_ema', data.get('G')) # Try 'G_ema' first, then 'G'
            if G is None:
                raise KeyError("Could not find 'G_ema' or 'G' in the network pickle.")
            G = G.eval().to(device)

    except Exception as e:
        logging.error(f"Failed to load StyleGAN model: {e}")
        raise
    # Disable gradients for inference
    for param in G.parameters():
        param.requires_grad = False
    logging.info("StyleGAN2 model loaded successfully and set to evaluation mode.")
    return G

def load_vit(model_name, pretrained, device):
    """Loads the ViT model and preprocessing from OpenCLIP."""
    logging.info(f"Loading OpenCLIP ViT model: {model_name} (pretrained: {pretrained})")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device # Load directly to the target device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        logging.error(f"Failed to load OpenCLIP model: {e}")
        raise
    logging.info("OpenCLIP ViT model loaded successfully and set to evaluation mode.")
    return model, preprocess, tokenizer

# -----------------------Image Generation----------------------
def generate_images_stylegan(G, num_images, batch_size, truncation_psi, device, output_dir, save_files=False):
    """Generates images using StyleGAN2 and optionally saves them."""
    logging.info(f"Starting image generation: {num_images} images, batch size {batch_size}.")
    start_time = time.time()

    all_latents_z = []
    all_latents_w = []
    generated_image_tensors = []
    image_paths = [] # Store paths if saving files

    G.eval() # Ensure model is in eval mode
    num_batches = (num_images + batch_size - 1) // batch_size # Calculate number of batches needed

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating Images"):
            current_batch_size = min(batch_size, num_images - i * batch_size)
            if current_batch_size <= 0:
                break

            # Generate latent vectors (z)
            z = torch.randn(current_batch_size, G.z_dim, device=device)
            all_latents_z.append(z.cpu()) # Store z latents on CPU

            # Map z to w space (optional truncation)
            # Note: Some StyleGAN versions might need G.mapping(z, None, truncation_psi=truncation_psi)
            # Check the specific legacy.py or model usage if errors occur.
            # Common usage:
            w = G.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=None) # Or specify cutoff if needed
            all_latents_w.append(w.cpu()) # Store w latents on CPU

            # Generate images from w
            generated_batch = G.synthesis(w, noise_mode='const')

            # Post-process for saving/display: [-1, 1] -> [0, 1]
            generated_batch_norm = (generated_batch.clamp(-1, 1) + 1) / 2.0
            generated_image_tensors.append(generated_batch_norm.cpu()) # Store tensors on CPU

            # Save images individually if requested
            if save_files:
                for j in range(current_batch_size):
                    img_index = i * batch_size + j
                    img_path = os.path.join(output_dir, f"img_{img_index:0{len(str(num_images))}d}.png")
                    save_image(generated_batch_norm[j], img_path)
                    image_paths.append(img_path)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_images if num_images > 0 else 0
    logging.info(f"Image generation finished. Generated {num_images} images in {total_time:.2f}s (Avg: {avg_time:.4f}s/image).")
    if save_files:
        logging.info(f"Images saved to: {output_dir}")

    # Concatenate lists of tensors into single tensors
    all_latents_z = torch.cat(all_latents_z, dim=0)
    all_latents_w = torch.cat(all_latents_w, dim=0)
    generated_image_tensors = torch.cat(generated_image_tensors, dim=0)

    return all_latents_z, all_latents_w, generated_image_tensors, image_paths

# -----------------------Feature Extraction----------------------
def extract_features_vit(vit_model, preprocess, image_tensors, batch_size, device):
    """Extracts features from image tensors using the ViT model."""
    num_images = image_tensors.shape[0]
    logging.info(f"Starting feature extraction for {num_images} images, batch size {batch_size}.")
    start_time = time.time()

    all_features = []
    vit_model.eval() # Ensure model is in eval mode
    num_batches = (num_images + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting Features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            batch_tensors = image_tensors[start_idx:end_idx]

            # Preprocess the batch
            # Apply the preprocessing steps provided by open_clip
            # Note: Ensure image_tensors are in [0, 1] range, which they should be from generation
            batch_processed = torch.stack([preprocess(T.ToPILImage()(img)) for img in batch_tensors]).to(device)

            # Use autocast for potential performance boost with mixed precision
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'): # MPS doesn't support autocast well
                batch_features = vit_model.encode_image(batch_processed)

            # Normalize features (L2 normalization)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features.cpu()) # Move features to CPU to free GPU memory

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Feature extraction finished in {total_time:.2f}s.")

    # Concatenate all features into a single tensor/numpy array
    image_features_tensor = torch.cat(all_features, dim=0)
    image_features_np = image_features_tensor.numpy()
    logging.info(f"Extracted features shape: {image_features_np.shape}")

    return image_features_np


# -----------------------Clustering and Visualization----------------------
def perform_clustering_and_visualization(features, n_clusters, use_pca, pca_components, tsne_perplexity, tsne_iterations, output_base_dir):
    """Performs K-means clustering and visualizes using t-SNE."""
    logging.info("Starting clustering and visualization.")
    vectors = features # Shape: (num_images, feature_dim)
    logging.info(f"Input features shape: {vectors.shape}")

    # --- K-means Clustering ---
    logging.info(f"Performing K-means clustering with {n_clusters} clusters.")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10) # n_init=10 is default and recommended
    cluster_labels = kmeans.fit_predict(vectors)
    logging.info(f"K-means clustering finished in {time.time() - start_time:.2f}s.")

    # --- Dimensionality Reduction for Visualization (t-SNE) ---
    features_for_tsne = vectors
    if use_pca:
        logging.info(f"Applying PCA to reduce dimensions to {pca_components} before t-SNE.")
        start_time = time.time()
        pca = PCA(n_components=pca_components, random_state=SEED)
        features_for_tsne = pca.fit_transform(features_for_tsne)
        logging.info(f"PCA finished in {time.time() - start_time:.2f}s.")
        logging.info(f"Shape after PCA: {features_for_tsne.shape}")
    else:
        logging.info("Skipping PCA before t-SNE.")

    logging.info("Applying t-SNE for 2D visualization.")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=tsne_perplexity, n_iter=tsne_iterations, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(features_for_tsne)
    logging.info(f"t-SNE finished in {time.time() - start_time:.2f}s.")
    logging.info(f"Shape after t-SNE: {vectors_2d.shape}")

    # --- Plotting t-SNE ---
    logging.info("Plotting t-SNE visualization.")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        vectors_2d[:, 0], vectors_2d[:, 1],
        c=cluster_labels, cmap='viridis', alpha=0.7, s=10 # Smaller points for potentially many samples
    )
    plt.title(f"t-SNE Visualization of Image Feature Clusters (k={n_clusters})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=range(n_clusters), title="Clusters")
    # cbar = plt.colorbar(scatter) # Use legend instead of colorbar for discrete labels
    # cbar.set_label("Cluster Label")
    plt.grid(True, linestyle='--', alpha=0.5)

    tsne_plot_path = os.path.join(output_base_dir, "tsne_clusters_visualization.png")
    plt.savefig(tsne_plot_path)
    logging.info(f"t-SNE plot saved to: {tsne_plot_path}")
    # plt.show() # Optionally display interactively
    plt.close() # Close figure to free memory

    return cluster_labels

# -----------------------Cluster Sample Visualization----------------------
def visualize_cluster_samples(G, latents_z, cluster_labels, n_clusters, n_samples, truncation_psi, device, output_base_dir):
    """Generates and saves sample images for each cluster."""
    logging.info("Generating sample images for each cluster.")

    # Group latents by cluster label
    latents_by_label = {label: latents_z[cluster_labels == label] for label in range(n_clusters)}

    for label, vecs in latents_by_label.items():
        if len(vecs) == 0:
            logging.warning(f"Cluster {label} has no samples.")
            continue

        logging.info(f"Generating {n_samples} samples for cluster {label} ({len(vecs)} items in cluster).")

        # Create a figure for the cluster samples
        num_cols = n_samples
        fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 3, 3.5))
        if num_cols == 1: axes = [axes] # Make sure axes is iterable even if n_samples=1

        fig.suptitle(f"Cluster {label} Samples", fontsize=16)

        samples_to_show = min(n_samples, len(vecs))
        indices = np.random.choice(len(vecs), samples_to_show, replace=False)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                z_sample = vecs[idx].unsqueeze(0).to(device) # Add batch dim and move to device

                # Generate image (using the same mapping and synthesis process)
                w_sample = G.mapping(z_sample, None, truncation_psi=truncation_psi, truncation_cutoff=None)
                img_tensor = G.synthesis(w_sample, noise_mode='const')

                # Post-process: [-1, 1] -> [0, 1] -> [0, 255] uint8 numpy
                img_norm = (img_tensor.clamp(-1, 1) + 1) / 2.0
                img_np = img_norm.squeeze(0).permute(1, 2, 0).cpu().numpy() # HWC format
                img_np = (img_np * 255).astype(np.uint8)

                # Display the image
                axes[i].imshow(img_np)
                axes[i].axis('off')
                axes[i].set_title(f"Sample {i+1}")

        # Hide any unused subplots if n_samples > len(vecs)
        for j in range(samples_to_show, num_cols):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        # Save the figure for this cluster
        cluster_plot_path = os.path.join(output_base_dir, f"cluster_{label}_samples.png")
        plt.savefig(cluster_plot_path)
        logging.info(f"Saved sample plot for cluster {label} to: {cluster_plot_path}")
        # plt.show() # Optionally display interactively
        plt.close(fig) # Close figure

# -----------------------Simple Generator (Optional)----------------------
# Define a simple generator network (as provided)
class SimpleGenerator(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=1024):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh() # Tanh squashes output to [-1, 1]. StyleGAN latents (z) are typically gaussian.
                       # If aiming to reproduce z, Tanh might not be ideal.
                       # If aiming to reproduce w, which might have different bounds, Tanh could be okay.
                       # Let's remove Tanh for z-space generation. Add if needed for w-space.
        )

    def forward(self, x):
        return self.model(x)

def train_simple_generator(latents_z, cluster_labels, target_cluster, n_epochs, batch_size, device, output_base_dir):
    """Trains the simple generator to map noise to latents of a specific cluster."""
    logging.info(f"--- Starting Simple Generator Training for Cluster {target_cluster} ---")

    # Select target latents
    target_latents = latents_z[cluster_labels == target_cluster]
    if len(target_latents) == 0:
        logging.error(f"Cannot train Simple Generator: Target cluster {target_cluster} has no samples.")
        return None

    target_latents = target_latents.to(device)
    logging.info(f"Training with {len(target_latents)} target latent vectors from cluster {target_cluster}.")

    # Instantiate the generator, optimizer, and loss
    latent_dim = latents_z.shape[1]
    simple_gen = SimpleGenerator(input_dim=latent_dim, output_dim=latent_dim).to(device)
    simple_gen.train() # Set to training mode
    optimizer = optim.Adam(simple_gen.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    logging.info("Starting training loop...")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        num_batches_train = (len(target_latents) + batch_size - 1) // batch_size

        # Simple approach: Try to map random noise to the *mean* or *random samples* of target latents
        # The original code compared noise output to the *entire* target set fixed, which is unusual.
        # Let's modify slightly: map noise -> target samples within the batch.
        
        pbar = tqdm(range(num_batches_train), desc=f"Epoch {epoch+1}/{n_epochs}")
        for i in pbar:
            optimizer.zero_grad()

            # Sample random noise input
            noise = torch.randn(batch_size, latent_dim, device=device)

            # Generate output vector (potential latent vector)
            generated_z = simple_gen(noise)

            # Select a batch of target latents to compare against
            # Simplest: sample randomly from the target set
            target_indices = torch.randint(0, len(target_latents), (batch_size,), device=device)
            batch_target_z = target_latents[target_indices]
            
            # Compute loss
            loss = criterion(generated_z, batch_target_z)

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / num_batches_train
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.6f}")

    logging.info("Simple Generator training finished.")
    # Save the trained generator model
    model_save_path = os.path.join(output_base_dir, f"simple_generator_cluster_{target_cluster}.pth")
    torch.save(simple_gen.state_dict(), model_save_path)
    logging.info(f"Simple Generator model saved to: {model_save_path}")

    return simple_gen

def evaluate_simple_generator(simple_gen, G_stylegan, n_samples, truncation_psi, device, output_base_dir, cluster_label):
    """Generates images using the trained SimpleGenerator."""
    if simple_gen is None:
        logging.warning("Simple Generator not available for evaluation.")
        return

    logging.info(f"Evaluating trained Simple Generator for cluster {cluster_label} by generating {n_samples} images.")
    simple_gen.eval() # Set simple generator to eval mode
    G_stylegan.eval() # Ensure StyleGAN is in eval mode

    latent_dim = simple_gen.model[0].in_features # Get latent dim

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3.5))
    if n_samples == 1: axes = [axes]
    fig.suptitle(f"Images from Trained Simple Generator (Target Cluster {cluster_label})", fontsize=16)

    with torch.no_grad():
        for i in range(n_samples):
            # Generate noise input
            noise = torch.randn(1, latent_dim, device=device)
            # Get latent vector from simple generator
            z_generated = simple_gen(noise)

            # Generate image using StyleGAN
            w_generated = G_stylegan.mapping(z_generated, None, truncation_psi=truncation_psi, truncation_cutoff=None)
            img_tensor = G_stylegan.synthesis(w_generated, noise_mode='const')

            # Post-process for display
            img_norm = (img_tensor.clamp(-1, 1) + 1) / 2.0
            img_np = img_norm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            axes[i].imshow(img_np)
            axes[i].axis('off')
            axes[i].set_title(f"Generated {i+1}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    eval_plot_path = os.path.join(output_base_dir, f"simple_generator_eval_cluster_{cluster_label}.png")
    plt.savefig(eval_plot_path)
    logging.info(f"Saved Simple Generator evaluation plot to: {eval_plot_path}")
    # plt.show()
    plt.close(fig)


# -----------------------Main Execution----------------------
if __name__ == "__main__":
    # --- 1. Setup ---
    setup_environment(SEED, LOG_FILE, OUTPUT_BASE_DIR, GENERATED_IMAGES_DIR)
    device = get_device(DEVICE_ID)

    # --- 2. Load Models ---
    G = load_stylegan(NETWORK_PKL, device)
    vit_model, vit_preprocess, _ = load_vit(VIT_MODEL_NAME, VIT_PRETRAINED, device)

    # --- 3. Generate Images and Latents ---
    latents_z, latents_w, generated_tensors, image_paths = generate_images_stylegan(
        G=G,
        num_images=NUM_IMAGES_TO_GENERATE,
        batch_size=GENERATION_BATCH_SIZE,
        truncation_psi=TRUNCATION_PSI,
        device=device,
        output_dir=GENERATED_IMAGES_DIR
        # save_files=SAVE_IMAGES
    )
    # generated_tensors are in [0, 1] range, suitable for ViT preprocess
    logging.info(f"Shape of generated tensors: {generated_tensors.shape}")
    logging.info(f"Shape of Z latents: {latents_z.shape}")
    logging.info(f"Shape of W latents: {latents_w.shape}")

    # Optional: Save latents if needed later (especially for large runs)
    latents_z_path = os.path.join(OUTPUT_BASE_DIR, "latents_z.npy")
    latents_w_path = os.path.join(OUTPUT_BASE_DIR, "latents_w.npy")
    np.save(latents_z_path, latents_z.numpy())
    np.save(latents_w_path, latents_w.numpy())
    logging.info(f"Z latents saved to {latents_z_path}")
    logging.info(f"W latents saved to {latents_w_path}")


    # --- 4. Extract Features ---
    image_features = extract_features_vit(
        vit_model=vit_model,
        preprocess=vit_preprocess,
        image_tensors=generated_tensors, # Pass the tensors directly
        batch_size=FEATURE_EXTRACTION_BATCH_SIZE,
        device=device
    )
    # Clear large tensor list from memory if not needed anymore
    del generated_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Optional: Save features
    features_path = os.path.join(OUTPUT_BASE_DIR, "image_features.npy")
    np.save(features_path, image_features)
    logging.info(f"Image features saved to {features_path}")


    # --- 5. Perform Clustering and Visualization ---
    cluster_labels = perform_clustering_and_visualization(
        features=image_features,
        n_clusters=NUM_CLUSTERS,
        use_pca=USE_PCA_BEFORE_TSNE,
        pca_components=PCA_COMPONENTS,
        tsne_perplexity=TSNE_PERPLEXITY,
        tsne_iterations=TSNE_ITERATIONS,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    # Optional: Save cluster labels
    labels_path = os.path.join(OUTPUT_BASE_DIR, "cluster_labels.npy")
    np.save(labels_path, cluster_labels)
    logging.info(f"Cluster labels saved to {labels_path}")


    # --- 6. Visualize Cluster Samples ---
    # We use the Z latents here, as they are the input to the mapping network
    visualize_cluster_samples(
        G=G,
        latents_z=latents_z,
        cluster_labels=cluster_labels,
        n_clusters=NUM_CLUSTERS,
        n_samples=CLUSTER_VIS_SAMPLES,
        truncation_psi=TRUNCATION_PSI,
        device=device,
        output_base_dir=OUTPUT_BASE_DIR
    )


    # --- 7. Train and Evaluate Simple Generator (Optional) ---
    if TRAIN_SIMPLE_GENERATOR:
        trained_simple_gen = train_simple_generator(
            latents_z=latents_z,
            cluster_labels=cluster_labels,
            target_cluster=SIMPLE_GENERATOR_TARGET_CLUSTER,
            n_epochs=SIMPLE_GENERATOR_EPOCHS,
            batch_size=SIMPLE_GENERATOR_BATCH_SIZE,
            device=device,
            output_base_dir=OUTPUT_BASE_DIR
        )

        evaluate_simple_generator(
            simple_gen=trained_simple_gen,
            G_stylegan=G,
            n_samples=5, # Generate 5 samples for evaluation
            truncation_psi=TRUNCATION_PSI,
            device=device,
            output_base_dir=OUTPUT_BASE_DIR,
            cluster_label=SIMPLE_GENERATOR_TARGET_CLUSTER
        )
    else:
        logging.info("Skipping Simple Generator training and evaluation.")

    logging.info("--- Experiment End ---")