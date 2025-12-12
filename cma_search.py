"""
CMA-ES Baseline: Novelty Search with Derivative-Free Optimization
==================================================================
This demonstrates that even sophisticated optimization on θ = (seed, cfg, steps)
cannot escape the semantic basin of a fixed prompt.

Key insight: CMA-ES is great at optimization, but the search space is wrong.
Language (prompts) is the key lever, not generation parameters.

Usage:
  python cma_search.py --iterations 100 --prompt "a painting of a landscape"
  
Requirements:
  pip install cma
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Warning: cma not installed. Run: pip install cma")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CMAConfig:
    # Model settings
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    embedding_type: str = "dino"  # "dino" or "clip"
    dino_model_id: str = "facebook/dinov2-base"
    clip_model_id: str = "ViT-L-14"
    
    # Fixed prompt (the whole point is to show this doesn't help)
    base_prompt: str = "a painting of a landscape"
    
    # Search space bounds for θ
    # We normalize to [0, 1] for CMA-ES, then denormalize
    seed_range: Tuple[int, int] = (0, 1_000_000)
    cfg_scale_range: Tuple[float, float] = (3.0, 15.0)
    steps_range: Tuple[int, int] = (15, 50)
    
    # CMA-ES settings
    iterations: int = 100
    population_size: int = 12  # λ (offspring per generation, 12 for A100)
    sigma0: float = 0.3  # Initial step size
    
    # Generation settings
    image_size: int = 512
    
    # Output settings
    output_dir: Path = Path("outputs/cma_search")
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)


# =============================================================================
# SEARCH SPACE MAPPING
# =============================================================================

def normalize_theta(theta_raw: dict, config: CMAConfig) -> np.ndarray:
    """Convert raw θ to normalized [0,1]^3 vector for CMA-ES."""
    seed_norm = (theta_raw["seed"] - config.seed_range[0]) / (config.seed_range[1] - config.seed_range[0])
    cfg_norm = (theta_raw["cfg_scale"] - config.cfg_scale_range[0]) / (config.cfg_scale_range[1] - config.cfg_scale_range[0])
    steps_norm = (theta_raw["steps"] - config.steps_range[0]) / (config.steps_range[1] - config.steps_range[0])
    return np.array([seed_norm, cfg_norm, steps_norm])


def denormalize_theta(x: np.ndarray, config: CMAConfig) -> dict:
    """Convert normalized vector back to raw θ."""
    # Clamp to [0, 1]
    x = np.clip(x, 0, 1)
    
    seed = int(x[0] * (config.seed_range[1] - config.seed_range[0]) + config.seed_range[0])
    cfg_scale = x[1] * (config.cfg_scale_range[1] - config.cfg_scale_range[0]) + config.cfg_scale_range[0]
    steps = int(x[2] * (config.steps_range[1] - config.steps_range[0]) + config.steps_range[0])
    
    return {
        "seed": seed,
        "cfg_scale": cfg_scale,
        "steps": steps,
    }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(config: CMAConfig):
    """Load required models."""
    from diffusers import StableDiffusionPipeline
    
    print("Loading models...")
    
    # Stable Diffusion
    print(f"  Loading Stable Diffusion from {config.sd_model_id}...")
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        config.sd_model_id,
        torch_dtype=config.dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    sd_pipe = sd_pipe.to(config.device)
    if hasattr(sd_pipe, 'enable_attention_slicing'):
        sd_pipe.enable_attention_slicing()
    
    # Embedding model
    if config.embedding_type == "dino":
        from transformers import AutoImageProcessor, AutoModel
        print(f"  Loading DINO from {config.dino_model_id}...")
        processor = AutoImageProcessor.from_pretrained(config.dino_model_id)
        model = AutoModel.from_pretrained(config.dino_model_id).to(config.device)
        model.eval()
    else:
        import open_clip
        print(f"  Loading CLIP {config.clip_model_id}...")
        model, _, processor = open_clip.create_model_and_transforms(
            config.clip_model_id,
            pretrained="openai",
            device=config.device,
        )
        model.eval()
    
    print("All models loaded!")
    return sd_pipe, model, processor


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def generate_image(sd_pipe, prompt: str, theta: dict, config: CMAConfig) -> Image.Image:
    """Generate image with Stable Diffusion."""
    generator = torch.Generator(device=config.device).manual_seed(theta["seed"])
    
    with torch.no_grad():
        result = sd_pipe(
            prompt=prompt,
            num_inference_steps=int(theta["steps"]),
            guidance_scale=float(theta["cfg_scale"]),
            generator=generator,
            height=config.image_size,
            width=config.image_size,
        )
    
    return result.images[0]


def generate_images_batch(sd_pipe, prompts: List[str], thetas: List[dict], config: CMAConfig) -> List[Image.Image]:
    """
    Generate a batch of images with Stable Diffusion.
    Note: SD pipeline requires same steps/cfg for batch, so we use the first theta's params.
    """
    # Create generators with different seeds
    generators = [torch.Generator(device=config.device).manual_seed(int(t["seed"])) for t in thetas]
    
    # Use first theta's steps and cfg (SD limitation for batching)
    steps = int(thetas[0]["steps"])
    cfg_scale = float(thetas[0]["cfg_scale"])
    
    with torch.no_grad():
        result = sd_pipe(
            prompt=prompts,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generators,
            height=config.image_size,
            width=config.image_size,
        )
    
    return result.images


def get_embeddings_batch(images: List[Image.Image], model, processor, config: CMAConfig) -> torch.Tensor:
    """Get normalized embeddings for a batch of images."""
    if config.embedding_type == "dino":
        inputs = processor(images=images, return_tensors="pt").to(config.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    else:
        images_tensor = torch.stack([processor(img) for img in images]).to(config.device)
        with torch.no_grad():
            embeddings = model.encode_image(images_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return embeddings


def get_embedding(image: Image.Image, model, processor, config: CMAConfig) -> torch.Tensor:
    """Get normalized embedding for an image."""
    if config.embedding_type == "dino":
        inputs = processor(images=image, return_tensors="pt").to(config.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    else:
        image_tensor = processor(image).unsqueeze(0).to(config.device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)


def compute_novelty(embedding: torch.Tensor, history: List[torch.Tensor], k: int = 10) -> float:
    """
    Compute k-NN novelty = mean cosine distance to k nearest neighbors.
    Paper standard: k=10
    """
    from novelty import compute_knn_novelty
    
    if len(history) == 0:
        return 1.0
    
    history_tensor = torch.stack(history)
    return compute_knn_novelty(embedding, history_tensor, k=k)


# =============================================================================
# CMA-ES SEARCH
# =============================================================================

@dataclass
class SearchResult:
    """Single search result."""
    iteration: int
    theta: dict
    novelty: float
    image_path: Optional[Path] = None
    embedding: Optional[torch.Tensor] = None  # For later analysis


def run_cma_search(config: CMAConfig):
    """Run CMA-ES novelty search."""
    if not HAS_CMA:
        raise ImportError("CMA-ES requires the 'cma' package. Install with: pip install cma")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CMA-ES NOVELTY SEARCH (Fixed Prompt Baseline)")
    print("=" * 70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Embedding: {config.embedding_type.upper()}")
    print(f"Prompt: {config.base_prompt}")
    print(f"Iterations: {config.iterations}")
    print(f"Population size: {config.population_size} (batch size)")
    print()
    
    # Load models
    sd_pipe, embedding_model, embedding_processor = load_models(config)
    
    # Initialize history and results
    history: List[torch.Tensor] = []
    results: List[SearchResult] = []
    novelty_history: List[float] = []
    
    # Initialize CMA-ES
    # Start from center of search space
    x0 = np.array([0.5, 0.5, 0.5])  # Normalized [seed, cfg, steps]
    
    # CMA-ES options
    opts = {
        'popsize': config.population_size,
        'maxiter': config.iterations // config.population_size,
        'verb_disp': 0,  # Suppress CMA output
        'bounds': [0, 1],  # Keep in [0, 1]^3
    }
    
    # Create optimizer
    # Note: CMA-ES minimizes, so we'll return -novelty
    es = cma.CMAEvolutionStrategy(x0, config.sigma0, opts)
    
    print("-" * 70)
    print("Starting CMA-ES search...")
    print("-" * 70)
    
    start_time = time.time()
    iteration = 0
    
    pbar = tqdm(total=config.iterations, desc="CMA-ES")
    
    while not es.stop() and iteration < config.iterations:
        # Get candidate solutions (batch)
        solutions = es.ask()
        
        # Limit batch to remaining iterations
        batch_size = min(len(solutions), config.iterations - iteration)
        solutions_batch = solutions[:batch_size]
        
        # Denormalize all solutions to get thetas
        thetas = [denormalize_theta(x, config) for x in solutions_batch]
        prompts = [config.base_prompt] * batch_size
        
        # BATCH: Generate all images at once
        images = generate_images_batch(sd_pipe, prompts, thetas, config)
        
        # BATCH: Get all embeddings at once
        embeddings = get_embeddings_batch(images, embedding_model, embedding_processor, config)
        
        fitnesses = []
        
        # Process results
        for j in range(batch_size):
            image = images[j]
            embedding = embeddings[j]
            theta = thetas[j]
            
            # Compute novelty
            novelty = compute_novelty(embedding, history)
            
            # Store result
            image_path = images_dir / f"img_{iteration:04d}_n{novelty:.3f}.png"
            image.save(image_path)
            
            result = SearchResult(
                iteration=iteration,
                theta={
                    "seed": int(theta["seed"]),
                    "cfg_scale": float(theta["cfg_scale"]),
                    "steps": int(theta["steps"]),
                },
                novelty=float(novelty),
                image_path=image_path,
                embedding=embedding.cpu(),  # Save for analysis
            )
            results.append(result)
            novelty_history.append(novelty)
            
            # Update history
            history.append(embedding)
            
            # CMA-ES minimizes, so return negative novelty
            fitnesses.append(-novelty)
            
            iteration += 1
            pbar.update(1)
        
        # Pad fitnesses if we processed fewer than asked
        while len(fitnesses) < len(solutions):
            fitnesses.append(0.0)  # Neutral fitness for unused solutions
        
        # Update CMA-ES with fitnesses
        es.tell(solutions, fitnesses)
        
        # Progress logging
        if iteration % 20 == 0:
            recent_novelty = np.mean(novelty_history[-20:]) if len(novelty_history) >= 20 else np.mean(novelty_history)
            tqdm.write(f"  [Iter {iteration}] Recent avg novelty: {recent_novelty:.4f}")
    
    pbar.close()
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print("Search complete!")
    print("-" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(results):.2f}s per sample)")
    
    # Statistics
    novelties = [r.novelty for r in results]
    print(f"\nNovelty Statistics:")
    print(f"  Min:    {min(novelties):.4f}")
    print(f"  Max:    {max(novelties):.4f}")
    print(f"  Mean:   {np.mean(novelties):.4f}")
    print(f"  Median: {np.median(novelties):.4f}")
    print(f"  Std:    {np.std(novelties):.4f}")
    
    # Save results
    save_results(results, novelty_history, config)
    
    print()
    print("=" * 70)
    print("CMA-ES SEARCH COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config.output_dir.absolute()}")
    
    return results, novelty_history


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_results(results: List[SearchResult], novelty_history: List[float], config: CMAConfig):
    """Save results and visualizations."""
    
    # 1. Novelty curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Novelty over iterations
    axes[0].plot(novelty_history, 'b-', alpha=0.5, linewidth=0.5)
    window = 10
    if len(novelty_history) > window:
        rolling = np.convolve(novelty_history, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(novelty_history)), rolling, 'r-', linewidth=2, label='Rolling avg')
    
    # Add trend line
    z = np.polyfit(range(len(novelty_history)), novelty_history, 1)
    p = np.poly1d(z)
    axes[0].plot(range(len(novelty_history)), p(range(len(novelty_history))), 
                 'g--', alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Novelty Score')
    axes[0].set_title(f'CMA-ES Novelty Search ({config.embedding_type.upper()})\nFixed prompt: "{config.base_prompt[:40]}..."')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(novelty_history, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(novelty_history), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(novelty_history):.4f}')
    axes[1].axvline(np.median(novelty_history), color='g', linestyle='--', 
                    label=f'Median: {np.median(novelty_history):.4f}')
    axes[1].set_xlabel('Novelty Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Novelty Scores')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "cma_novelty_curve.png", dpi=150)
    plt.close()
    
    # 2. Gallery of top novel images
    sorted_results = sorted(results, key=lambda r: r.novelty, reverse=True)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f"CMA-ES Top 16 Novel Images\n(Fixed prompt, optimized θ)", fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(sorted_results):
            r = sorted_results[idx]
            if r.image_path and r.image_path.exists():
                img = Image.open(r.image_path)
                ax.imshow(img)
                ax.set_title(f"N={r.novelty:.3f}\ncfg={r.theta['cfg_scale']:.1f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "cma_gallery.png", dpi=150)
    plt.close()
    
    # 2b. Gallery of BOTTOM (least novel) images for comparison
    bottom_results = sorted_results[-16:][::-1]  # Least novel, reversed for display
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f"CMA-ES Bottom 16 (Least Novel) Images\n(Typical/Expected outputs)", fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(bottom_results):
            r = bottom_results[idx]
            if r.image_path and r.image_path.exists():
                img = Image.open(r.image_path)
                ax.imshow(img)
                ax.set_title(f"N={r.novelty:.3f}\ncfg={r.theta['cfg_scale']:.1f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "cma_gallery_bottom.png", dpi=150)
    plt.close()
    
    # 3. Save JSON log
    log_data = {
        "config": {
            "prompt": str(config.base_prompt),
            "embedding_type": str(config.embedding_type),
            "iterations": int(config.iterations),
            "population_size": int(config.population_size),
        },
        "stats": {
            "mean_novelty": float(np.mean(novelty_history)),
            "median_novelty": float(np.median(novelty_history)),
            "max_novelty": float(max(novelty_history)),
            "min_novelty": float(min(novelty_history)),
            "std_novelty": float(np.std(novelty_history)),
        },
        "results": [
            {
                "iteration": int(r.iteration),
                "theta": {
                    "seed": int(r.theta["seed"]),
                    "cfg_scale": float(r.theta["cfg_scale"]),
                    "steps": int(r.theta["steps"]),
                },
                "novelty": float(r.novelty),
                "image_path": str(r.image_path) if r.image_path else None,
            }
            for r in results
        ]
    }
    
    with open(config.output_dir / "cma_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Save embeddings for later analysis
    embeddings_list = [r.embedding.numpy() for r in results if r.embedding is not None]
    if embeddings_list:
        embeddings_array = np.stack(embeddings_list)
        np.save(config.output_dir / "embeddings.npy", embeddings_array)
        print(f"  Saved embeddings: {embeddings_array.shape}")
    
    print(f"  Saved: cma_novelty_curve.png, cma_gallery.png, cma_log.json")


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_methods(random_log: Path, cma_log: Path, output_path: Path):
    """
    Generate comparison visualization between random search and CMA-ES.
    Both should have been run with the same fixed prompt.
    """
    # Load logs
    with open(random_log) as f:
        random_data = json.load(f)
    with open(cma_log) as f:
        cma_data = json.load(f)
    
    random_novelties = [r["novelty"] for r in random_data["results"]]
    cma_novelties = [r["novelty"] for r in cma_data["results"]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Novelty curves
    ax1 = axes[0]
    ax1.plot(random_novelties, 'b-', alpha=0.5, label='Random Search')
    ax1.plot(cma_novelties, 'r-', alpha=0.5, label='CMA-ES')
    
    # Rolling averages
    window = 10
    if len(random_novelties) > window:
        random_rolling = np.convolve(random_novelties, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(random_novelties)), random_rolling, 'b-', linewidth=2)
    if len(cma_novelties) > window:
        cma_rolling = np.convolve(cma_novelties, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(cma_novelties)), cma_rolling, 'r-', linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Novelty')
    ax1.set_title('Random vs CMA-ES Novelty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distributions
    ax2 = axes[1]
    ax2.hist(random_novelties, bins=20, alpha=0.5, label=f'Random (μ={np.mean(random_novelties):.3f})')
    ax2.hist(cma_novelties, bins=20, alpha=0.5, label=f'CMA-ES (μ={np.mean(cma_novelties):.3f})')
    ax2.set_xlabel('Novelty')
    ax2.set_ylabel('Count')
    ax2.set_title('Novelty Distributions')
    ax2.legend()
    
    # 3. Summary stats
    ax3 = axes[2]
    methods = ['Random', 'CMA-ES']
    means = [np.mean(random_novelties), np.mean(cma_novelties)]
    stds = [np.std(random_novelties), np.std(cma_novelties)]
    
    x = np.arange(len(methods))
    bars = ax3.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel('Mean Novelty')
    ax3.set_title('Comparison: Both Still Collapse\n(Fixed prompt limits exploration)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Annotate
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{mean:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Comparison saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMA-ES Novelty Search (Fixed Prompt Baseline)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations")
    parser.add_argument("--prompt", type=str, 
                        default="a painting of a landscape",
                        help="Fixed prompt for generation")
    parser.add_argument("--embedding", type=str, default="dino",
                        choices=["dino", "clip"],
                        help="Embedding model to use")
    parser.add_argument("--population", type=int, default=8,
                        help="CMA-ES population size")
    parser.add_argument("--output-dir", type=str, default="outputs/cma_search",
                        help="Output directory")
    
    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare random vs CMA-ES (requires both logs)")
    parser.add_argument("--random-log", type=str,
                        help="Path to random search log.json")
    parser.add_argument("--cma-log", type=str,
                        help="Path to CMA search log.json")
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.random_log or not args.cma_log:
            print("Error: --compare requires --random-log and --cma-log")
            return
        compare_methods(
            Path(args.random_log),
            Path(args.cma_log),
            Path(args.output_dir) / "random_vs_cma.png"
        )
        return
    
    config = CMAConfig(
        iterations=args.iterations,
        base_prompt=args.prompt,
        embedding_type=args.embedding,
        population_size=args.population,
        output_dir=Path(args.output_dir),
    )
    
    run_cma_search(config)


if __name__ == "__main__":
    main()