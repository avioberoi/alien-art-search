"""
Phase 1: Alien Art via Novelty Search in CLIP Space
====================================================
Core loop:
1. Sample random θ = (seed, cfg_scale, steps)
2. Generate image with Stable Diffusion
3. Embed with CLIP
4. Score novelty = 1 - max(similarity to history)
5. Log and repeat

Run: python search.py --num_samples 100 --prompt "a painting of a landscape"
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel


# Config
@dataclass
class SearchConfig:
    # Model settings
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    clip_model_id: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    embedding_type: str = "clip"  # "clip" or "dino"
    dino_model_id: str = "facebook/dinov2-base"
    
    # Search space bounds
    seed_range: tuple = (0, 1_000_000)
    cfg_scale_range: tuple = (3.0, 15.0)
    steps_options: list = field(default_factory=lambda: [20, 25, 30, 40])
    
    # Generation settings
    base_prompt: str = "a painting of a landscape"
    image_size: int = 512
    
    # Search settings
    num_samples: int = 100
    batch_size: int = 8
    
    # Output settings
    output_dir: Path = Path("outputs/search")
    save_all_images: bool = False  # Set True to save every image
    save_top_k: int = 16  # Save top K most novel
    save_bottom_k: int = 16  # Save bottom K least novel
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)


# Sampling
def sample_theta(config: SearchConfig, rng: np.random.Generator) -> dict:
    """Sample random pt in search space"""
    return {
        "seed": int(rng.integers(config.seed_range[0], config.seed_range[1])),
        "cfg_scale": float(rng.uniform(config.cfg_scale_range[0], config.cfg_scale_range[1])),
        "steps": int(rng.choice(config.steps_options)),
    }


# Model Loading
def load_stable_diffusion(config: SearchConfig):
    """Load Stable Diffusion pipeline"""
    from diffusers import StableDiffusionPipeline
    
    print(f"Loading Stable Diffusion from {config.sd_model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.sd_model_id,
        torch_dtype=config.dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device)
    
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    
    print("Stable Diffusion loaded successfully")
    return pipe


def load_clip(config: SearchConfig):
    """Load CLIP model & preprocessing"""
    import open_clip
    
    print(f"Loading CLIP {config.clip_model_id}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model_id,
        pretrained=config.clip_pretrained,
        device=config.device,
    )
    model.eval()
    
    print("CLIP loaded successfully")
    return model, preprocess


def load_dino(config: SearchConfig):
    """Load DINO model & processor"""
    print(f"Loading DINO from {config.dino_model_id}...")
    processor = AutoImageProcessor.from_pretrained(config.dino_model_id)
    model = AutoModel.from_pretrained(config.dino_model_id)
    model.to(config.device)
    model.eval()
    
    print("DINO loaded successfully")
    return model, processor


# Core Functions
def generate_images(pipe, prompts: list[str], theta: dict, generators: list[torch.Generator], config: SearchConfig) -> list[Image.Image]:
    """Generate batch of images"""
    with torch.no_grad():
        result = pipe(
            prompt=prompts,
            num_inference_steps=theta["steps"],
            guidance_scale=theta["cfg_scale"],
            generator=generators,
            height=config.image_size,
            width=config.image_size,
        )
    
    return result.images


def get_clip_embeddings(images: list[Image.Image], clip_model, preprocess, 
                       config: SearchConfig) -> torch.Tensor:
    """Get normalized CLIP embeddings for batch of images"""
    # Preprocess all images and stack
    images_tensor = torch.stack([preprocess(img) for img in images]).to(config.device)
    
    with torch.no_grad():
        embeddings = clip_model.encode_image(images_tensor)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return embeddings


def get_dino_embeddings(images: list[Image.Image], model, processor, 
                       config: SearchConfig) -> torch.Tensor:
    """Get normalized DINO embeddings for batch of images"""
    inputs = processor(images=images, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token (first token) for embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return embeddings


def compute_novelty_batch(embeddings: torch.Tensor, history: list[torch.Tensor], k: int = 10) -> list[float]:
    """
    Compute k-NN novelty score for batch of embeddings given history
    Novelty = mean cosine distance to k nearest neighbors
    Paper std: k=10
    """
    from novelty import compute_knn_novelty_batch
    
    if len(history) == 0:
        return [1.0] * embeddings.shape[0]
    
    history_tensor = torch.stack(history)
    return compute_knn_novelty_batch(embeddings, history_tensor, k=k)


# Logging and Visualization
@dataclass
class SearchResult:
    """Single search result"""
    index: int
    theta: dict
    novelty: float
    image_path: Optional[Path] = None
    embedding: Optional[torch.Tensor] = None


def save_search_log(results: list[SearchResult], config: SearchConfig):
    """Save search results to JSON"""
    log_path = config.output_dir / "search_log.json"
    
    log_data = {
        "config": {
            "base_prompt": config.base_prompt,
            "num_samples": config.num_samples,
            "seed_range": config.seed_range,
            "cfg_scale_range": config.cfg_scale_range,
            "steps_options": config.steps_options,
        },
        "results": [
            {
                "index": r.index,
                "theta": r.theta,
                "novelty": r.novelty,
                "image_path": str(r.image_path) if r.image_path else None,
            }
            for r in results
        ]
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Search log saved to {log_path}")


def plot_novelty_curve(results: list[SearchResult], config: SearchConfig):
    """Plot novelty over iterations"""
    novelties = [r.novelty for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Novelty over iterations
    ax1 = axes[0]
    ax1.plot(novelties, 'b-', alpha=0.7, linewidth=1)
    ax1.scatter(range(len(novelties)), novelties, c=novelties, cmap='viridis', 
                s=20, alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Novelty Score')
    ax1.set_title('Novelty vs Iteration')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(novelties)), novelties, 1)
    p = np.poly1d(z)
    ax1.plot(range(len(novelties)), p(range(len(novelties))), 
             'r--', alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
    ax1.legend()
    
    # Plot 2: Histogram of novelty scores
    ax2 = axes[1]
    ax2.hist(novelties, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(novelties), color='r', linestyle='--', 
                label=f'Mean: {np.mean(novelties):.4f}')
    ax2.axvline(np.median(novelties), color='g', linestyle='--', 
                label=f'Median: {np.median(novelties):.4f}')
    ax2.set_xlabel('Novelty Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Novelty Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = config.output_dir / "novelty_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Novelty plot saved to {plot_path}")


def create_gallery(results: list[SearchResult], config: SearchConfig):
    """Create comparison gallery of high vs low novelty images"""
    # Sort by novelty
    sorted_results = sorted(results, key=lambda r: r.novelty, reverse=True)
    
    # Get top and bottom K
    top_k = sorted_results[:config.save_top_k]
    bottom_k = sorted_results[-config.save_bottom_k:]
    
    # Create grid for high novelty
    create_image_grid(
        [r for r in top_k if r.image_path and r.image_path.exists()],
        config.output_dir / "gallery_high_novelty.png",
        title="High Novelty (Most Alien)"
    )
    
    # Create grid for low novelty
    create_image_grid(
        [r for r in bottom_k if r.image_path and r.image_path.exists()],
        config.output_dir / "gallery_low_novelty.png",
        title="Low Novelty (Most Typical)"
    )


def create_image_grid(results: list[SearchResult], save_path: Path, 
                      title: str, grid_size: int = 4):
    """Create grid of images with novelty scores as captions"""
    n = min(len(results), grid_size * grid_size)
    if n == 0:
        return
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            r = results[idx]
            img = Image.open(r.image_path)
            ax.imshow(img)
            ax.set_title(f"N={r.novelty:.3f}\ncfg={r.theta['cfg_scale']:.1f}", 
                        fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gallery saved to {save_path}")


# Main Search Loop
def run_search(config: SearchConfig):
    """Run novelty search"""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ALIEN ART - NOVELTY SEARCH IN FM SPACE")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Samples: {config.num_samples}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Prompt: {config.base_prompt}")
    print(f"Output: {config.output_dir}")
    print()
    
    # Load models
    sd_pipe = load_stable_diffusion(config)
    
    if config.embedding_type == "clip":
        embedding_model, embedding_processor = load_clip(config)
    elif config.embedding_type == "dino":
        embedding_model, embedding_processor = load_dino(config)
    else:
        raise ValueError(f"Unknown embedding type: {config.embedding_type}")
    
    # Initialize
    rng = np.random.default_rng(seed=42)
    history: list[torch.Tensor] = []
    results: list[SearchResult] = []
    
    print()
    print("-" * 70)
    print("Starting search...")
    print("-" * 70)
    
    start_time = time.time()
    
    # Iterate in batches
    pbar = tqdm(total=config.num_samples, desc="Searching")
    
    for i in range(0, config.num_samples, config.batch_size):
        # Adjust batch size for last batch
        current_batch_size = min(config.batch_size, config.num_samples - i)
        
        # Sample shared parameters for the batch
        # We sample one set of (cfg, steps) but unique seeds
        base_theta = sample_theta(config, rng)
        batch_seeds = [int(rng.integers(config.seed_range[0], config.seed_range[1])) for _ in range(current_batch_size)]
        
        # Prepare batch arguments
        prompts = [config.base_prompt] * current_batch_size
        generators = [torch.Generator(device=config.device).manual_seed(s) for s in batch_seeds]
        
        # Generate batch
        images = generate_images(sd_pipe, prompts, base_theta, generators, config)
        
        # Get embeddings
        if config.embedding_type == "clip":
            embeddings = get_clip_embeddings(images, embedding_model, embedding_processor, config)
        elif config.embedding_type == "dino":
            embeddings = get_dino_embeddings(images, embedding_model, embedding_processor, config)
        
        # Compute novelty
        novelties = compute_novelty_batch(embeddings, history)
        
        # Process batch results
        for j in range(current_batch_size):
            global_idx = i + j
            img = images[j]
            novelty = novelties[j]
            embedding = embeddings[j]
            seed = batch_seeds[j]
            
            # Construct theta for this specific sample
            theta = base_theta.copy()
            theta["seed"] = seed
            
            # Save image
            image_path = images_dir / f"img_{global_idx:04d}_n{novelty:.3f}.png"
            img.save(image_path)
            
            # Create result
            result = SearchResult(
                index=global_idx,
                theta=theta,
                novelty=novelty,
                image_path=image_path,
                embedding=embedding.cpu(),
            )
            results.append(result)
        
        # Update history with new embeddings
        history.extend([e for e in embeddings])
        
        # Update progress
        pbar.update(current_batch_size)
        
        # Log recent novelty
        if (i // config.batch_size + 1) % 5 == 0:
            recent_novelty = np.mean([r.novelty for r in results[-10:]])
            tqdm.write(f"  [Batch {i//config.batch_size + 1}] Recent avg novelty: {recent_novelty:.4f}")
            
    pbar.close()
    
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print("Search complete!")
    print("-" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/config.num_samples:.2f}s per sample)")
    
    # Statistics
    novelties = [r.novelty for r in results]
    print(f"\nNovelty Statistics:")
    print(f"  Min:    {min(novelties):.4f}")
    print(f"  Max:    {max(novelties):.4f}")
    print(f"  Mean:   {np.mean(novelties):.4f}")
    print(f"  Median: {np.median(novelties):.4f}")
    print(f"  Std:    {np.std(novelties):.4f}")
    
    # Save results
    print()
    print("Saving results...")
    save_search_log(results, config)
    plot_novelty_curve(results, config)
    create_gallery(results, config)
    
    # Save embeddings for later analysis (k-NN visualization)
    embeddings_array = np.stack([r.embedding.numpy() for r in results])
    np.save(config.output_dir / "embeddings.npy", embeddings_array)
    print(f"  Saved embeddings: {embeddings_array.shape}")
    
    print()
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config.output_dir.absolute()}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Alien Art Novelty Search")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--prompt", type=str,
                        default="a painting of a landscape",
                        help="Base prompt for generation")
    parser.add_argument("--output_dir", type=str, default="outputs/search",
                        help="Output directory")
    parser.add_argument("--cfg_min", type=float, default=3.0,
                        help="Minimum CFG scale")
    parser.add_argument("--cfg_max", type=float, default=15.0,
                        help="Maximum CFG scale")
    parser.add_argument("--embedding_type", type=str, default="clip", choices=["clip", "dino"],
                        help="Embedding model type: clip or dino")
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-base",
                        help="DINO model ID (if embedding_type is dino)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = SearchConfig(
        num_samples=args.num_samples,
        base_prompt=args.prompt,
        output_dir=Path(args.output_dir),
        cfg_scale_range=(args.cfg_min, args.cfg_max),
        embedding_type=args.embedding_type,
        dino_model_id=args.dino_model,
    )
    
    run_search(config)


if __name__ == "__main__":
    main()