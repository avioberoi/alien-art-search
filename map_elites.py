"""
MAP-Elites for Alien Art Discovery
===================================
Quality-Diversity search in prompt space with DINO evaluation.
Supports both predefined semantic axes and automatic PCA-based grids.

Usage:
  # With predefined axes (interactive demo)
  python map_elites.py --axis1 "abstract,figurative" --axis2 "organic,geometric" --iterations 200
  
  # With automatic PCA
  python map_elites.py --auto-axes --iterations 200
  
  # Quick test
  python map_elites.py --iterations 50 --grid-size 5
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import json
import time
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from prompts import (
    generate_random_prompt, 
    mutate_prompt, 
    SEED_PROMPTS,
    generate_prompt_batch,
)
from prompts_constrained import (
    mutate_art_prompt,
    SEED_PROMPTS_CONSTRAINED,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MAPElitesConfig:
    # Model settings
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    dino_model_id: str = "facebook/dinov2-base"
    clip_model_id: str = "ViT-L-14"  # For semantic axis measurement
    
    # Embedding model for novelty computation ("dino" or "clip")
    embedding_model: str = "dino"
    
    # Generation settings
    image_size: int = 512
    cfg_scale_range: Tuple[float, float] = (5.0, 12.0)
    steps_options: List[int] = field(default_factory=lambda: [20, 25, 30])
    
    # MAP-Elites settings
    grid_size: int = 10  # 10x10 grid = 100 cells
    iterations: int = 200
    batch_size: int = 8  # Batch size for generation (Increased for GPU utilization)
    
    # Semantic axes (for predefined mode)
    # Format: (concept_low, concept_high) for each axis
    axis1: Optional[Tuple[str, str]] = None  # e.g., ("abstract art", "realistic photograph")
    axis2: Optional[Tuple[str, str]] = None  # e.g., ("organic shapes", "geometric patterns")
    
    # Auto mode: use PCA on DINO embeddings
    auto_axes: bool = False
    
    # Art-constrained mode: only use art-style prompts (paintings, art movements)
    use_art_prompts: bool = False
    
    # Output settings
    output_dir: Path = Path("outputs/map_elites")
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)


# =============================================================================
# PREDEFINED SEMANTIC AXES
# =============================================================================

# Common axis definitions for interactive demos
AXIS_PRESETS = {
    # Visual style axes
    "abstract_figurative": ("pure abstract art, non-representational", "realistic figurative art, representational"),
    "organic_geometric": ("organic natural shapes, flowing curves", "geometric angular shapes, mathematical patterns"),
    
    # Complexity axes
    "simple_complex": ("minimalist simple composition", "intricate complex detailed composition"),
    "chaotic_ordered": ("chaotic random disordered", "ordered structured symmetrical"),
    
    # Mood/feeling axes
    "alien_familiar": ("alien strange unfamiliar otherworldly", "familiar earthly recognizable natural"),
    "dark_light": ("dark ominous mysterious shadowy", "bright luminous radiant ethereal"),
    
    # Content axes
    "micro_macro": ("microscopic cellular molecular", "cosmic astronomical galactic vast"),
    "ancient_futuristic": ("ancient primitive archaeological", "futuristic technological advanced"),
    
    # Material axes
    "organic_synthetic": ("biological organic living natural", "mechanical synthetic artificial metallic"),
    "solid_ethereal": ("solid dense heavy material", "ethereal gaseous translucent immaterial"),
    
    # WOW-FACTOR axes - for unexpected juxtapositions
    "mundane_extraordinary": ("mundane ordinary everyday boring familiar", "extraordinary magical impossible transcendent alien"),
    "comforting_unsettling": ("comforting safe warm cozy familiar", "unsettling disturbing uncanny liminal eerie"),
    "expected_surprising": ("expected predictable conventional normal", "surprising unexpected jarring incongruous weird"),
}


def parse_axis(axis_str: str) -> Tuple[str, str]:
    """Parse axis string like 'abstract,figurative' or use preset."""
    if axis_str in AXIS_PRESETS:
        return AXIS_PRESETS[axis_str]
    
    if "," in axis_str:
        parts = axis_str.split(",", 1)
        return (parts[0].strip(), parts[1].strip())
    
    raise ValueError(f"Invalid axis format: {axis_str}. Use 'concept1,concept2' or a preset name.")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(config: MAPElitesConfig):
    """Load all required models."""
    from diffusers import StableDiffusionPipeline
    import open_clip
    from build_wikiart_reference import ImageEncoder
    
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
    
    # Primary embedder for novelty (DINO or CLIP based on config)
    print(f"  Loading {config.embedding_model.upper()} embedder for novelty...")
    novelty_embedder = ImageEncoder(
        model_type=config.embedding_model,
        device=config.device,
        dino_model_id=config.dino_model_id,
        clip_model_id=config.clip_model_id,
    )
    
    # DINO (for PCA-based auto-axes, always load for backward compat)
    if config.embedding_model != "dino":
        print(f"  Loading DINO from {config.dino_model_id} (for auto-axes)...")
        from transformers import AutoImageProcessor, AutoModel
        dino_processor = AutoImageProcessor.from_pretrained(config.dino_model_id)
        dino_model = AutoModel.from_pretrained(config.dino_model_id).to(config.device)
        dino_model.eval()
    else:
        # Use the novelty embedder's model directly
        dino_model = novelty_embedder.model
        dino_processor = novelty_embedder.processor
    
    # CLIP (for semantic axes)
    print(f"  Loading CLIP {config.clip_model_id}...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        config.clip_model_id,
        pretrained="openai",
        device=config.device,
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(config.clip_model_id)
    
    print("All models loaded!")
    
    return {
        "sd": sd_pipe,
        "dino": (dino_model, dino_processor),
        "clip": (clip_model, clip_preprocess, tokenizer),
        "novelty_embedder": novelty_embedder,
    }


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_dino_embedding(image: Image.Image, dino_model, dino_processor, device: str) -> torch.Tensor:
    """Get normalized DINO embedding for an image."""
    inputs = dino_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)


def get_clip_image_embedding(image: Image.Image, clip_model, clip_preprocess, device: str) -> torch.Tensor:
    """Get normalized CLIP image embedding."""
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)


def get_clip_text_embedding(text: str, clip_model, tokenizer, device: str) -> torch.Tensor:
    """Get normalized CLIP text embedding."""
    tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        embedding = clip_model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)


# =============================================================================
# CELL COMPUTATION (Behavior Characterization)
# =============================================================================

class SemanticAxes:
    """Compute cell position based on semantic axes."""
    
    def __init__(self, axis1: Tuple[str, str], axis2: Tuple[str, str], 
                 clip_model, tokenizer, device: str, grid_size: int):
        self.grid_size = grid_size
        self.device = device
        self.clip_model = clip_model
        
        # Precompute axis embeddings
        print(f"  Axis 1: '{axis1[0]}' ←→ '{axis1[1]}'")
        print(f"  Axis 2: '{axis2[0]}' ←→ '{axis2[1]}'")
        
        self.axis1_low = get_clip_text_embedding(axis1[0], clip_model, tokenizer, device)
        self.axis1_high = get_clip_text_embedding(axis1[1], clip_model, tokenizer, device)
        self.axis2_low = get_clip_text_embedding(axis2[0], clip_model, tokenizer, device)
        self.axis2_high = get_clip_text_embedding(axis2[1], clip_model, tokenizer, device)
        
        # Axis directions
        self.axis1_dir = self.axis1_high - self.axis1_low
        self.axis2_dir = self.axis2_high - self.axis2_low
        
        # Normalize
        self.axis1_dir = self.axis1_dir / self.axis1_dir.norm()
        self.axis2_dir = self.axis2_dir / self.axis2_dir.norm()
        
        self.axis1_labels = axis1
        self.axis2_labels = axis2
    
    def get_position(self, clip_embedding: torch.Tensor) -> Tuple[float, float]:
        """Get continuous position along both axes."""
        # Project onto axis directions
        pos1 = (clip_embedding @ self.axis1_dir).item()
        pos2 = (clip_embedding @ self.axis2_dir).item()
        return (pos1, pos2)
    
    def get_cell(self, clip_embedding: torch.Tensor) -> Tuple[int, int]:
        """Get discrete cell position."""
        pos1, pos2 = self.get_position(clip_embedding)
        
        # Normalize to [0, 1] range (roughly)
        # These are cosine similarities, typically in [-0.5, 0.5] range
        norm1 = (pos1 + 0.5) / 1.0  # Shift and scale
        norm2 = (pos2 + 0.5) / 1.0
        
        # Clamp and discretize
        cell1 = int(np.clip(norm1 * self.grid_size, 0, self.grid_size - 1))
        cell2 = int(np.clip(norm2 * self.grid_size, 0, self.grid_size - 1))
        
        return (cell1, cell2)


class AutoAxes:
    """Compute cell position using PCA on DINO embeddings."""
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.pca = None
        self.embeddings_buffer = []
        self.min_vals = None
        self.max_vals = None
    
    def update_pca(self, embeddings: List[torch.Tensor]):
        """Update PCA with accumulated embeddings."""
        if len(embeddings) < 10:
            return  # Need minimum samples
        
        # Stack embeddings
        X = torch.stack(embeddings).cpu().numpy()
        
        # Fit PCA
        self.pca = PCA(n_components=2)
        transformed = self.pca.fit_transform(X)
        
        # Update normalization bounds
        self.min_vals = transformed.min(axis=0)
        self.max_vals = transformed.max(axis=0)
    
    def get_position(self, dino_embedding: torch.Tensor) -> Optional[Tuple[float, float]]:
        """Get continuous position in PCA space."""
        if self.pca is None:
            return None
        
        X = dino_embedding.cpu().numpy().reshape(1, -1)
        pos = self.pca.transform(X)[0]
        return (pos[0], pos[1])
    
    def get_cell(self, dino_embedding: torch.Tensor) -> Tuple[int, int]:
        """Get discrete cell position."""
        pos = self.get_position(dino_embedding)
        
        if pos is None or self.min_vals is None:
            # Fallback: random cell until PCA is fitted
            return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        
        # Normalize to [0, 1]
        range_vals = self.max_vals - self.min_vals + 1e-8
        norm = (np.array(pos) - self.min_vals) / range_vals
        
        # Discretize
        cell1 = int(np.clip(norm[0] * self.grid_size, 0, self.grid_size - 1))
        cell2 = int(np.clip(norm[1] * self.grid_size, 0, self.grid_size - 1))
        
        return (cell1, cell2)


# =============================================================================
# ARCHIVE (Elite Storage)
# =============================================================================

@dataclass
class Elite:
    """A single elite in the archive."""
    prompt: str
    image_path: Path
    novelty_embedding: torch.Tensor  # Primary embedding for novelty (DINO or CLIP)
    clip_embedding: torch.Tensor  # Always store CLIP for semantic axes
    novelty: float
    cell: Tuple[int, int]
    iteration: int
    theta: dict  # seed, cfg_scale, steps
    embedding_type: str = "dino"  # Track which type was used


class Archive:
    """MAP-Elites archive."""
    
    def __init__(self, grid_size: int, embedding_type: str = "dino"):
        self.grid_size = grid_size
        self.embedding_type = embedding_type
        self.elites: Dict[Tuple[int, int], Elite] = {}
        self.all_novelty_embeddings: List[torch.Tensor] = []
    
    def compute_novelty(self, novelty_embedding: torch.Tensor, k: int = 10) -> float:
        """
        Compute k-NN novelty against archive.
        Paper standard: k=10
        """
        from novelty import compute_knn_novelty
        
        if len(self.all_novelty_embeddings) == 0:
            return 1.0
        
        archive_matrix = torch.stack(self.all_novelty_embeddings)
        return compute_knn_novelty(novelty_embedding, archive_matrix, k=k)
    
    def try_insert(self, elite: Elite) -> bool:
        """Try to insert elite. Returns True if inserted."""
        cell = elite.cell
        
        # Compute novelty against current archive
        elite.novelty = self.compute_novelty(elite.novelty_embedding)
        
        if cell not in self.elites:
            # Empty cell - insert
            self.elites[cell] = elite
            self.all_novelty_embeddings.append(elite.novelty_embedding)
            return True
        
        # Cell occupied - compare novelty (or could use other quality metric)
        existing = self.elites[cell]
        if elite.novelty > existing.novelty:
            # Replace with better elite
            # Remove old embedding
            # (Note: this is approximate - we keep all embeddings for novelty computation)
            self.elites[cell] = elite
            self.all_novelty_embeddings.append(elite.novelty_embedding)
            return True
        
        return False
    
    def select_parent(self) -> Optional[Elite]:
        """Select a random elite as parent for mutation."""
        if len(self.elites) == 0:
            return None
        return np.random.choice(list(self.elites.values()))
    
    def coverage(self) -> float:
        """Fraction of cells filled."""
        return len(self.elites) / (self.grid_size ** 2)
    
    def get_top_novel(self, k: int = 10) -> List[Elite]:
        """Get top-k most novel elites."""
        sorted_elites = sorted(self.elites.values(), key=lambda e: e.novelty, reverse=True)
        return sorted_elites[:k]


# =============================================================================
# GENERATION
# =============================================================================

def generate_images(sd_pipe, prompts: List[str], thetas: List[dict], config: MAPElitesConfig) -> List[Image.Image]:
    """Generate a batch of images with Stable Diffusion."""
    # We use the first seed for the generator, but SD pipeline handles batch seeds differently
    # Ideally we should pass a list of generators, one for each prompt
    generators = [torch.Generator(device=config.device).manual_seed(t["seed"]) for t in thetas]
    
    # Use the first theta for shared parameters (steps, cfg) - assuming batch shares these or we take average/first
    # In this implementation, we'll assume the batch shares the same steps/cfg for efficiency if possible,
    # OR we have to run them sequentially if they differ.
    # However, standard SD pipeline takes single steps/guidance_scale.
    # To support varying parameters in a batch, we might need to group by params or run sequentially.
    # For simplicity in this optimization, let's assume we sample ONE set of (steps, cfg) for the whole batch
    # but different seeds/prompts.
    
    # If we want fully independent parameters, we can't easily batch with standard pipeline without custom loop.
    # Let's stick to: Batch of prompts, Batch of seeds, Shared steps/CFG for this batch.
    
    steps = int(thetas[0]["steps"])  # Convert numpy.int64 to int
    cfg = float(thetas[0]["cfg_scale"])  # Ensure float

    # Get negative prompt if specified - must be list to match prompts type
    neg_prompt_str = thetas[0].get("negative_prompt", None)
    if neg_prompt_str is not None:
        # Convert numpy.str_ to str and replicate for batch
        negative_prompt = [str(neg_prompt_str)] * len(prompts)
    else:
        negative_prompt = None
    
    with torch.no_grad():
        result = sd_pipe(
            prompt=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generators,
            height=config.image_size,
            width=config.image_size,
        )
    
    return result.images


# Default negative prompts for quality
DEFAULT_NEGATIVE_PROMPTS = [
    "blurry, low quality, text, watermark, deformed",
    "blurry, low quality, text, watermark, ugly",
    "blurry, distorted, low resolution, artifacts",
]

# Creative negative prompts for pushing boundaries
CREATIVE_NEGATIVE_PROMPTS = [
    "realistic, photographic, normal, ordinary, familiar",
    "symmetrical, repetitive, predictable, conventional",
    "human faces, recognizable objects, earthly, mundane",
    "straight lines, geometric, structured, organized",
]

# Anti-cliche negative prompts - block expected sci-fi/alien aesthetics for wow-factor
ANTI_CLICHE_NEGATIVE_PROMPTS = [
    "sci-fi, space, cosmic, nebula, galaxy, stars",
    "tentacles, biomechanical, Giger, alien creature",
    "purple glow, blue glow, neon, futuristic lighting",
    "crystal, crystalline, gems, jewels, faceted",
    "portal, vortex, wormhole, dimensional rift",
    "typical alien, expected weird, obvious surreal",
]

def sample_theta(config: MAPElitesConfig, use_creative_negatives: bool = False) -> dict:
    """Sample random generation parameters."""
    theta = {
        "seed": int(np.random.randint(0, 1_000_000)),  # Convert to native int for JSON
        "cfg_scale": float(np.random.uniform(*config.cfg_scale_range)),  # Convert to native float
        "steps": int(np.random.choice(config.steps_options)),  # Convert to native int
    }
    
    # Optionally add negative prompt
    if use_creative_negatives:
        r = np.random.random()
        if r < 0.25:
            # 25% chance of creative negative prompt
            theta["negative_prompt"] = str(np.random.choice(CREATIVE_NEGATIVE_PROMPTS))
        elif r < 0.45:
            # 20% chance of anti-cliche negative (blocks expected sci-fi aesthetics)
            theta["negative_prompt"] = str(np.random.choice(ANTI_CLICHE_NEGATIVE_PROMPTS))
        elif r < 0.75:
            # 30% chance of quality negative prompt
            theta["negative_prompt"] = str(np.random.choice(DEFAULT_NEGATIVE_PROMPTS))
        # 25% chance of no negative prompt
    elif np.random.random() < 0.5:
        # 50% chance of quality negative prompt
        theta["negative_prompt"] = str(np.random.choice(DEFAULT_NEGATIVE_PROMPTS))
    
    return theta


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_map_elites(config: MAPElitesConfig):
    """Run MAP-Elites search."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("MAP-ELITES: ALIEN ART DISCOVERY")
    print("=" * 70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Grid size: {config.grid_size}x{config.grid_size} ({config.grid_size**2} cells)")
    print(f"Iterations: {config.iterations}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Output: {config.output_dir}")
    print()
    
    # Load models
    models = load_models(config)
    sd_pipe = models["sd"]
    dino_model, dino_processor = models["dino"]
    clip_model, clip_preprocess, tokenizer = models["clip"]
    novelty_embedder = models["novelty_embedder"]
    
    # Initialize axes
    print("\nInitializing behavior axes...")
    if config.auto_axes:
        print("  Mode: Automatic (PCA on DINO embeddings)")
        axes = AutoAxes(config.grid_size)
    else:
        if config.axis1 is None:
            config.axis1 = AXIS_PRESETS["abstract_figurative"]
        if config.axis2 is None:
            config.axis2 = AXIS_PRESETS["organic_geometric"]
        print("  Mode: Predefined semantic axes")
        axes = SemanticAxes(
            config.axis1, config.axis2,
            clip_model, tokenizer, config.device, config.grid_size
        )
    
    # Initialize archive
    archive = Archive(config.grid_size)
    
    # Tracking
    novelty_history = []
    coverage_history = []
    
    print()
    print("-" * 70)
    print("Starting search...")
    print("-" * 70)
    
    start_time = time.time()
    
    # Process in batches
    pbar = tqdm(total=config.iterations, desc="MAP-Elites")
    
    # Select seed prompts and mutation function based on mode
    if config.use_art_prompts:
        seed_prompts = SEED_PROMPTS_CONSTRAINED
        mutation_fn = mutate_art_prompt
        print("  Prompt Mode: Art-constrained (paintings, art styles)")
    else:
        seed_prompts = SEED_PROMPTS
        mutation_fn = mutate_prompt
        print("  Prompt Mode: Unconstrained (any subject)")
    
    for i in range(0, config.iterations, config.batch_size):
        current_batch_size = min(config.batch_size, config.iterations - i)
        
        batch_prompts = []
        batch_thetas = []
        
        # 1. Prepare batch
        for _ in range(current_batch_size):
            # Select parent or generate random prompt
            if len(archive.elites) > 0 and np.random.random() < 0.8:
                # 80% of time: mutate from archive
                parent = archive.select_parent()
                prompt = mutation_fn(parent.prompt)
            else:
                # 20% of time: fresh random prompt
                if len(batch_prompts) + i < len(seed_prompts):
                    prompt = seed_prompts[len(batch_prompts) + i]
                else:
                    if config.use_art_prompts:
                        # Generate new art-style prompt
                        prompt = mutation_fn(random.choice(seed_prompts))
                    else:
                        prompt = generate_random_prompt()
            
            batch_prompts.append(prompt)
            
            # Sample theta
            # Note: For batch efficiency, we might want to force shared steps/cfg
            # But let's sample independently and then override for the batch to match the first one
            # This is a trade-off for speed.
            theta = sample_theta(config, use_creative_negatives=True)
            batch_thetas.append(theta)
            
        # Enforce shared steps/cfg for the batch to allow parallel generation
        # (Stable Diffusion pipeline limitation for batching)
        shared_steps = batch_thetas[0]["steps"]
        shared_cfg = batch_thetas[0]["cfg_scale"]
        for t in batch_thetas:
            t["steps"] = shared_steps
            t["cfg_scale"] = shared_cfg
        
        # 2. Generate batch
        images = generate_images(sd_pipe, batch_prompts, batch_thetas, config)
        
        # 3. Process results
        for j, image in enumerate(images):
            global_idx = i + j
            prompt = batch_prompts[j]
            theta = batch_thetas[j]
            
            # Get embeddings
            # Use novelty_embedder for primary embedding (DINO or CLIP based on config)
            novelty_emb = novelty_embedder.encode_single(image)
            clip_emb = get_clip_image_embedding(image, clip_model, clip_preprocess, config.device)
            
            # For auto-axes, we need DINO embedding for PCA
            if config.auto_axes and config.embedding_model != "dino":
                dino_emb = get_dino_embedding(image, dino_model, dino_processor, config.device)
            else:
                dino_emb = novelty_emb if config.embedding_model == "dino" else novelty_emb
            
            # Get cell position
            if config.auto_axes:
                # Update PCA periodically
                if len(archive.all_novelty_embeddings) > 0 and global_idx % 20 == 0:
                    axes.update_pca([e for e in archive.all_novelty_embeddings])
                cell = axes.get_cell(dino_emb)
            else:
                cell = axes.get_cell(clip_emb)
            
            # Save image
            image_path = images_dir / f"img_{global_idx:04d}.png"
            image.save(image_path)
            
            # Create elite
            elite = Elite(
                prompt=prompt,
                image_path=image_path,
                novelty_embedding=novelty_emb.cpu(),
                clip_embedding=clip_emb.cpu(),
                novelty=0.0,  # Will be computed in try_insert
                cell=cell,
                iteration=global_idx,
                theta=theta,
                embedding_type=config.embedding_model,
            )
            
            # Try to insert
            inserted = archive.try_insert(elite)
            
            # Track metrics
            novelty_history.append(elite.novelty)
            coverage_history.append(archive.coverage())
        
        # Update progress
        pbar.update(current_batch_size)
        
        # Progress logging
        if (i // config.batch_size + 1) % 5 == 0:
            tqdm.write(
                f"  [Batch {i//config.batch_size + 1}] Coverage: {archive.coverage()*100:.1f}% "
                f"({len(archive.elites)}/{config.grid_size**2} cells), "
                f"Recent novelty: {np.mean(novelty_history[-20:]):.3f}"
            )
            
    pbar.close()
    
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print("Search complete!")
    print("-" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/config.iterations:.2f}s per iteration)")
    print(f"Final coverage: {archive.coverage()*100:.1f}% ({len(archive.elites)}/{config.grid_size**2} cells)")
    
    # Statistics
    all_novelties = [e.novelty for e in archive.elites.values()]
    print(f"\nArchive Novelty Statistics:")
    print(f"  Min:    {min(all_novelties):.4f}")
    print(f"  Max:    {max(all_novelties):.4f}")
    print(f"  Mean:   {np.mean(all_novelties):.4f}")
    print(f"  Median: {np.median(all_novelties):.4f}")
    
    # Save results
    print("\nSaving results...")
    save_results(archive, novelty_history, coverage_history, config, axes)
    
    print()
    print("=" * 70)
    print("MAP-ELITES COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config.output_dir.absolute()}")
    
    return archive, novelty_history, coverage_history


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_results(archive: Archive, novelty_history: List[float], 
                 coverage_history: List[float], config: MAPElitesConfig,
                 axes):
    """Save all results and visualizations."""
    
    # 1. Novelty and coverage curves
    fig, axes_plot = plt.subplots(1, 2, figsize=(14, 5))
    
    axes_plot[0].plot(novelty_history, 'b-', alpha=0.5, linewidth=0.5)
    # Rolling average
    window = 20
    if len(novelty_history) > window:
        rolling = np.convolve(novelty_history, np.ones(window)/window, mode='valid')
        axes_plot[0].plot(range(window-1, len(novelty_history)), rolling, 'r-', linewidth=2, label='Rolling avg')
    axes_plot[0].set_xlabel('Iteration')
    axes_plot[0].set_ylabel('Novelty Score')
    axes_plot[0].set_title('Novelty vs Iteration')
    axes_plot[0].legend()
    axes_plot[0].grid(True, alpha=0.3)
    
    axes_plot[1].plot(coverage_history, 'g-', linewidth=2)
    axes_plot[1].set_xlabel('Iteration')
    axes_plot[1].set_ylabel('Coverage (%)')
    axes_plot[1].set_title('Archive Coverage vs Iteration')
    axes_plot[1].grid(True, alpha=0.3)
    axes_plot[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "metrics.png", dpi=150)
    plt.close()
    
    # 2. Archive grid visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create grid
    grid_img = np.ones((config.grid_size * 64, config.grid_size * 64, 3), dtype=np.uint8) * 200
    
    for (x, y), elite in archive.elites.items():
        # Load and resize image
        img = Image.open(elite.image_path).resize((64, 64))
        img_array = np.array(img)
        
        # Place in grid
        grid_img[y*64:(y+1)*64, x*64:(x+1)*64] = img_array[:, :, :3]
    
    ax.imshow(grid_img)
    
    # Add axis labels if semantic axes
    if hasattr(axes, 'axis1_labels'):
        ax.set_xlabel(f"← {axes.axis1_labels[0]}  |  {axes.axis1_labels[1]} →", fontsize=12)
        ax.set_ylabel(f"← {axes.axis2_labels[0]}  |  {axes.axis2_labels[1]} →", fontsize=12)
    else:
        ax.set_xlabel("PCA Dimension 1", fontsize=12)
        ax.set_ylabel("PCA Dimension 2", fontsize=12)
    
    ax.set_title(f"MAP-Elites Archive ({len(archive.elites)}/{config.grid_size**2} cells filled)", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "archive_grid.png", dpi=150)
    plt.close()
    
    # 3. Top novel gallery (ALIEN)
    top_novel = archive.get_top_novel(16)
    
    fig, axes_gallery = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Top 16 Most Novel Images (ALIEN Gallery)", fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes_gallery.flat):
        if idx < len(top_novel):
            elite = top_novel[idx]
            img = Image.open(elite.image_path)
            ax.imshow(img)
            ax.set_title(f"N={elite.novelty:.3f}\n{elite.prompt[:40]}...", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "alien_gallery.png", dpi=150)
    plt.close()
    
    # 4. Bottom novel gallery (TYPICAL - for reference)
    sorted_elites = sorted(archive.elites.values(), key=lambda e: e.novelty)
    bottom_novel = sorted_elites[:16]
    
    fig, axes_gallery = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Bottom 16 Least Novel Images (TYPICAL Gallery - Reference)", fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes_gallery.flat):
        if idx < len(bottom_novel):
            elite = bottom_novel[idx]
            img = Image.open(elite.image_path)
            ax.imshow(img)
            ax.set_title(f"N={elite.novelty:.3f}\n{elite.prompt[:40]}...", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "typical_gallery.png", dpi=150)
    plt.close()
    
    # 5. Side-by-side comparison
    fig, axes_compare = plt.subplots(2, 8, figsize=(24, 8))
    fig.suptitle("Alien vs Typical: Why Novelty Matters", fontsize=16, fontweight='bold')
    
    # Top row: Most novel
    for idx in range(8):
        ax = axes_compare[0, idx]
        if idx < len(top_novel):
            elite = top_novel[idx]
            img = Image.open(elite.image_path)
            ax.imshow(img)
            ax.set_title(f"N={elite.novelty:.3f}", fontsize=10, color='green')
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel("ALIEN\n(High Novelty)", fontsize=12, fontweight='bold')
    
    # Bottom row: Least novel
    for idx in range(8):
        ax = axes_compare[1, idx]
        if idx < len(bottom_novel):
            elite = bottom_novel[idx]
            img = Image.open(elite.image_path)
            ax.imshow(img)
            ax.set_title(f"N={elite.novelty:.3f}", fontsize=10, color='red')
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel("TYPICAL\n(Low Novelty)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "alien_vs_typical.png", dpi=150)
    plt.close()
    
    # 4. Save archive data as JSON
    archive_data = {
        "config": {
            "grid_size": int(config.grid_size),
            "iterations": int(config.iterations),
            "auto_axes": bool(config.auto_axes),
            "use_art_prompts": bool(config.use_art_prompts),
            "embedding_model": str(config.embedding_model),
        },
        "stats": {
            "coverage": float(archive.coverage()),
            "num_elites": int(len(archive.elites)),
            "novelty_mean": float(np.mean([e.novelty for e in archive.elites.values()])),
        },
        "elites": [
            {
                "cell": [int(c) for c in elite.cell],  # Convert tuple of numpy.int64 to list of int
                "prompt": str(elite.prompt),
                "novelty": float(elite.novelty),
                "iteration": int(elite.iteration),
                "theta": {k: (int(v) if isinstance(v, (int, np.integer)) else 
                              float(v) if isinstance(v, (float, np.floating)) else 
                              str(v) if v is not None else None)
                          for k, v in elite.theta.items()},
                "image_path": str(elite.image_path),
            }
            for elite in archive.elites.values()
        ]
    }
    
    with open(config.output_dir / "archive_data.json", 'w') as f:
        json.dump(archive_data, f, indent=2)
    
    # Save embeddings for later analysis
    if archive.all_novelty_embeddings:
        embeddings_array = torch.stack(archive.all_novelty_embeddings).numpy()
        np.save(config.output_dir / "embeddings.npy", embeddings_array)
        print(f"  Saved {config.embedding_model.upper()} embeddings: {embeddings_array.shape}")
    
    print(f"  Saved: metrics.png, archive_grid.png, alien_gallery.png, typical_gallery.png, alien_vs_typical.png, archive_data.json")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MAP-Elites Alien Art Discovery")
    
    # Core settings
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of iterations")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Grid size (NxN cells)")
    parser.add_argument("--output-dir", type=str, default="outputs/map_elites",
                        help="Output directory")
    
    # Axis mode
    parser.add_argument("--auto-axes", action="store_true",
                        help="Use automatic PCA axes instead of semantic")
    parser.add_argument("--axis1", type=str, default=None,
                        help="First semantic axis: 'concept1,concept2' or preset name")
    parser.add_argument("--axis2", type=str, default=None,
                        help="Second semantic axis: 'concept1,concept2' or preset name")
    
    # Prompt mode
    parser.add_argument("--use-art-prompts", action="store_true",
                        help="Use art-constrained prompts (paintings, art styles) instead of general prompts")
    
    # Embedding model for novelty
    parser.add_argument("--embedding-model", type=str, default="dino", choices=["dino", "clip"],
                        help="Embedding model for novelty computation (dino or clip)")
    
    # List presets
    parser.add_argument("--list-presets", action="store_true",
                        help="List available axis presets and exit")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available axis presets:")
        for name, (low, high) in AXIS_PRESETS.items():
            print(f"  {name}:")
            print(f"    Low:  {low}")
            print(f"    High: {high}")
        return
    
    config = MAPElitesConfig(
        iterations=args.iterations,
        grid_size=args.grid_size,
        output_dir=Path(args.output_dir),
        auto_axes=args.auto_axes,
        use_art_prompts=args.use_art_prompts,
        embedding_model=args.embedding_model,
    )
    
    # Parse axes if provided
    if args.axis1:
        config.axis1 = parse_axis(args.axis1)
    if args.axis2:
        config.axis2 = parse_axis(args.axis2)
    
    run_map_elites(config)


if __name__ == "__main__":
    main()