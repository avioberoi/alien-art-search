"""
Phase 0: Alien Art Pipeline - Demo Script
==========================================
Validates that we can:
1. Generate images with Stable Diffusion
2. Embed them with CLIP
3. Compute pairwise cosine similarities

Run: python demo.py
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class Config:
    # Model settings
    sd_model_id = "runwayml/stable-diffusion-v1-5"  # Lighter than SDXL
    clip_model_id = "ViT-L-14"
    clip_pretrained = "openai"
    
    # Generation settings
    base_prompt = "a painting of a landscape"
    num_inference_steps = 25
    guidance_scale = 7.5
    image_size = 512
    
    # Demo settings
    num_demo_images = 4
    output_dir = Path("outputs/demo")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_stable_diffusion(config: Config):
    """Load Stable Diffusion pipeline."""
    from diffusers import StableDiffusionPipeline
    
    print(f"Loading Stable Diffusion from {config.sd_model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.sd_model_id,
        torch_dtype=config.dtype,
        safety_checker=None,  # Disable for speed
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device)
    
    # Enable memory optimizations
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    
    print("Stable Diffusion loaded successfully.")
    return pipe


def load_clip(config: Config):
    """Load CLIP model and preprocessing."""
    import open_clip
    
    print(f"Loading CLIP {config.clip_model_id}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model_id,
        pretrained=config.clip_pretrained,
        device=config.device,
    )
    model.eval()
    
    print("CLIP loaded successfully.")
    return model, preprocess


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def generate_image(pipe, prompt: str, seed: int, cfg_scale: float, 
                   steps: int, config: Config) -> Image.Image:
    """Generate a single image with specified parameters."""
    generator = torch.Generator(device=config.device).manual_seed(seed)
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            height=config.image_size,
            width=config.image_size,
        )
    
    return result.images[0]


def get_clip_embedding(image: Image.Image, clip_model, preprocess, 
                       config: Config) -> torch.Tensor:
    """Get normalized CLIP embedding for an image."""
    # Preprocess image
    image_tensor = preprocess(image).unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        # L2 normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)


def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    return (z1 @ z2).item()


def compute_similarity_matrix(embeddings: list[torch.Tensor]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    
    return sim_matrix


# -----------------------------------------------------------------------------
# Main Demo
# -----------------------------------------------------------------------------

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ALIEN ART PIPELINE - PHASE 0 DEMO")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    print()
    
    # Load models
    sd_pipe = load_stable_diffusion(config)
    clip_model, clip_preprocess = load_clip(config)
    
    print()
    print("-" * 60)
    print("Generating demo images...")
    print("-" * 60)
    
    # Generate images with different seeds
    images = []
    embeddings = []
    seeds = [42, 123, 456, 789][:config.num_demo_images]
    
    for i, seed in enumerate(tqdm(seeds, desc="Generating")):
        start_time = time.time()
        
        # Generate image
        image = generate_image(
            pipe=sd_pipe,
            prompt=config.base_prompt,
            seed=seed,
            cfg_scale=config.guidance_scale,
            steps=config.num_inference_steps,
            config=config,
        )
        
        # Get embedding
        embedding = get_clip_embedding(image, clip_model, clip_preprocess, config)
        
        # Save image
        image_path = config.output_dir / f"demo_{i}_seed{seed}.png"
        image.save(image_path)
        
        images.append(image)
        embeddings.append(embedding)
        
        elapsed = time.time() - start_time
        print(f"  Image {i+1}: seed={seed}, time={elapsed:.2f}s, saved to {image_path}")
    
    print()
    print("-" * 60)
    print("Computing pairwise similarities...")
    print("-" * 60)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)
    
    print("\nPairwise Cosine Similarity Matrix:")
    print("(Higher = more similar, 1.0 = identical)")
    print()
    
    # Print header
    header = "        " + "  ".join([f"Seed {s:4d}" for s in seeds])
    print(header)
    print("-" * len(header))
    
    # Print matrix
    for i, seed in enumerate(seeds):
        row = f"Seed {seed:4d}  "
        row += "  ".join([f"{sim_matrix[i, j]:.4f}  " for j in range(len(seeds))])
        print(row)
    
    # Compute novelty scores (distance from nearest neighbor)
    print()
    print("-" * 60)
    print("Novelty Scores (1 - max similarity to others):")
    print("-" * 60)
    
    for i, seed in enumerate(seeds):
        # Get max similarity to OTHER images
        sims_to_others = [sim_matrix[i, j] for j in range(len(seeds)) if j != i]
        max_sim = max(sims_to_others)
        novelty = 1.0 - max_sim
        print(f"  Seed {seed}: novelty = {novelty:.4f} (max_sim = {max_sim:.4f})")
    
    print()
    print("=" * 60)
    print("PHASE 0 COMPLETE - Pipeline is working!")
    print("=" * 60)
    print(f"\nImages saved to: {config.output_dir.absolute()}")
    
    return sim_matrix, embeddings, images


if __name__ == "__main__":
    main()