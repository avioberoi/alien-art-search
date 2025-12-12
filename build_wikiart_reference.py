"""
Build WikiArt Reference Cloud with Elbow Analysis
==================================================
Scales up WikiArt reference from HuggingFace (81K images).
Includes elbow analysis to find optimal reference size.
Caches images locally for nearest-neighbor visualization.

Usage:
  python build_wikiart_reference.py --num-samples 20000 --cache-images
  python build_wikiart_reference.py --elbow-analysis  # Find optimal size
"""

import torch
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Paths
    output_dir: Path = Path("/project/jevans/avi/wikiart_reference")
    cache_dir: Path = Path("/project/jevans/avi/wikiart_cache")
    hf_cache: Path = Path("/project/jevans/avi/hf_cache")
    
    # Model - now supports both DINO and CLIP
    embedding_model: str = "dino"  # "dino" or "clip"
    dino_model_id: str = "facebook/dinov2-base"
    clip_model_id: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    
    # Reference settings
    num_samples: int = 20000
    batch_size: int = 64  # 64 for A100 (embedding only, no generation)
    
    # Elbow analysis
    elbow_checkpoints: List[int] = None  # Set in __post_init__
    
    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        if self.elbow_checkpoints is None:
            self.elbow_checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 81000]


# =============================================================================
# UNIFIED IMAGE ENCODER (DINO or CLIP)
# =============================================================================

class ImageEncoder:
    """
    Unified image encoder supporting both DINO and CLIP.
    This is the single source of truth for image embeddings across the pipeline.
    """
    
    def __init__(self, model_type: str = "dino", device: str = "cuda", 
                 dino_model_id: str = "facebook/dinov2-base",
                 clip_model_id: str = "ViT-L-14", 
                 clip_pretrained: str = "openai",
                 hf_cache: Path = None):
        self.model_type = model_type.lower()
        self.device = device
        self.model = None
        self.processor = None
        self.dimension = None
        
        if self.model_type == "dino":
            self._load_dino(dino_model_id, hf_cache)
        elif self.model_type == "clip":
            self._load_clip(clip_model_id, clip_pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'dino' or 'clip'.")
    
    def _load_dino(self, model_id: str, hf_cache: Path = None):
        """Load DINOv2 model."""
        from transformers import AutoImageProcessor, AutoModel
        
        print(f"Loading DINO ({model_id})...")
        cache_dir = str(hf_cache) if hf_cache else None
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        self.dimension = 768  # DINOv2-base
        
        # Try torch.compile
        try:
            self.model = torch.compile(self.model)
            print("  DINO compiled with torch.compile")
        except Exception:
            pass
        
        print(f"  DINO loaded. Embedding dim: {self.dimension}")
    
    def _load_clip(self, model_id: str, pretrained: str):
        """Load CLIP model."""
        import open_clip
        
        print(f"Loading CLIP ({model_id}, {pretrained})...")
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        
        # Get actual dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_out = self.model.encode_image(dummy)
            self.dimension = dummy_out.shape[-1]
        
        print(f"  CLIP loaded. Embedding dim: {self.dimension}")
    
    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode batch of images to normalized embeddings."""
        if self.model_type == "dino":
            return self._encode_dino(images)
        else:
            return self._encode_clip(images)
    
    def _encode_dino(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode with DINO."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()
    
    def _encode_clip(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode with CLIP."""
        batch_tensors = torch.stack([self.processor(img) for img in images]).to(self.device)
        embeddings = self.model.encode_image(batch_tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().float()  # CLIP may return float16
    
    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """Encode a single image."""
        return self.encode([image]).squeeze(0)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text (CLIP only)."""
        if self.model_type != "clip":
            raise ValueError("Text encoding only supported for CLIP")
        
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze(0).cpu().float()


# Backward compatibility alias
DINOEncoder = lambda config: ImageEncoder(
    model_type="dino", 
    device=config.device,
    dino_model_id=config.dino_model_id,
    hf_cache=config.hf_cache
)


# =============================================================================
# BUILD REFERENCE
# =============================================================================

def build_wikiart_reference(
    config: Config,
    cache_images: bool = False,
    save_metadata: bool = True,
) -> Dict:
    """
    Build WikiArt reference cloud with full metadata.
    Supports both DINO and CLIP embeddings based on config.embedding_model.
    
    Returns dict with:
      - embeddings: [N, D] tensor (D=768 for DINO, 768 for CLIP ViT-L-14)
      - metadata: list of dicts with style, artist, genre, image_path
    """
    from datasets import load_dataset
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if cache_images:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"BUILDING WIKIART REFERENCE CLOUD ({config.embedding_model.upper()})")
    print("=" * 70)
    print(f"Embedding model: {config.embedding_model}")
    print(f"Target samples: {config.num_samples}")
    print(f"Output: {config.output_dir}")
    if cache_images:
        print(f"Image cache: {config.cache_dir}")
    
    # Load dataset (streaming for memory efficiency)
    print("\nLoading huggan/wikiart dataset...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    # Initialize encoder (DINO or CLIP based on config)
    encoder = ImageEncoder(
        model_type=config.embedding_model,
        device=config.device,
        dino_model_id=config.dino_model_id,
        clip_model_id=config.clip_model_id,
        clip_pretrained=config.clip_pretrained,
        hf_cache=config.hf_cache,
    )
    
    # Process images
    embeddings = []
    metadata_list = []
    batch_images = []
    batch_metadata = []
    
    print(f"\nProcessing {config.num_samples} images...")
    pbar = tqdm(total=config.num_samples, desc="Embedding")
    
    for i, sample in enumerate(ds):
        if len(embeddings) >= config.num_samples:
            break
        
        try:
            image = sample["image"].convert('RGB')
            
            # Collect metadata
            meta = {
                "idx": len(embeddings) + len(batch_images),
                "style": sample.get("style", "unknown"),
                "artist": sample.get("artist", "unknown"),
                "genre": sample.get("genre", "unknown"),
            }
            
            # Cache image if requested
            if cache_images:
                img_filename = f"{meta['idx']:06d}.jpg"
                img_path = config.cache_dir / img_filename
                image.save(img_path, quality=85)
                meta["image_path"] = str(img_path)
            
            batch_images.append(image)
            batch_metadata.append(meta)
            
            # Process batch
            if len(batch_images) >= config.batch_size:
                emb = encoder.encode(batch_images)
                embeddings.append(emb)
                metadata_list.extend(batch_metadata)
                pbar.update(len(batch_images))
                batch_images = []
                batch_metadata = []
                
        except Exception as e:
            continue
    
    # Process remaining
    if batch_images:
        emb = encoder.encode(batch_images)
        embeddings.append(emb)
        metadata_list.extend(batch_metadata)
        pbar.update(len(batch_images))
    
    pbar.close()
    
    # Combine embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"\nEmbedded {len(all_embeddings)} images")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Compute statistics
    styles = {}
    for m in metadata_list:
        s = m["style"]
        styles[s] = styles.get(s, 0) + 1
    
    print(f"Styles represented: {len(styles)}")
    print("Top 10 styles:")
    for style, count in sorted(styles.items(), key=lambda x: -x[1])[:10]:
        print(f"  {style}: {count}")
    
    # Save reference - use embedding model in filename
    emb_type = config.embedding_model.lower()
    reference = {
        "embeddings": all_embeddings,
        "source": "wikiart_hf",
        "embedding_type": emb_type,
        "embedding_dim": all_embeddings.shape[1],
        "num_samples": len(all_embeddings),
        "metadata": metadata_list if save_metadata else None,
        "style_distribution": styles,
    }
    
    output_path = config.output_dir / f"wikiart_{emb_type}_{len(all_embeddings)}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(reference, f)
    print(f"\nSaved reference to {output_path}")
    
    # Also build and save FAISS index for efficient k-NN
    try:
        import faiss
        print("\nBuilding FAISS index for efficient k-NN search...")
        emb_np = all_embeddings.numpy().astype(np.float32)
        index = faiss.IndexFlatIP(emb_np.shape[1])  # Inner product for cosine sim
        index.add(emb_np)
        faiss_path = config.output_dir / f"wikiart_{emb_type}_{len(all_embeddings)}.faiss"
        faiss.write_index(index, str(faiss_path))
        print(f"Saved FAISS index to {faiss_path}")
    except ImportError:
        print("FAISS not available, skipping index creation")
    except Exception as e:
        print(f"FAISS index creation failed: {e}")
    
    return reference


# =============================================================================
# ELBOW ANALYSIS
# =============================================================================

def elbow_analysis(config: Config) -> Dict:
    """
    Analyze how embedding distribution stabilizes with sample size.
    
    Metrics computed at each checkpoint:
      - Mean pairwise distance (coverage)
      - Std of pairwise distances (uniformity)
      - Mean k-NN distance (density)
    """
    from datasets import load_dataset
    from scipy.spatial.distance import cdist
    
    print("=" * 70)
    print("ELBOW ANALYSIS: Finding Optimal WikiArt Reference Size")
    print("=" * 70)
    print(f"Checkpoints: {config.elbow_checkpoints}")
    
    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    # Initialize encoder
    encoder = DINOEncoder(config)
    
    # Process up to max checkpoint
    max_samples = max(config.elbow_checkpoints)
    embeddings = []
    batch_images = []
    
    print(f"\nProcessing up to {max_samples} images...")
    pbar = tqdm(total=max_samples, desc="Embedding")
    
    results = {"checkpoints": [], "metrics": []}
    checkpoint_idx = 0
    
    for i, sample in enumerate(ds):
        if len(embeddings) * config.batch_size + len(batch_images) >= max_samples:
            break
        
        try:
            image = sample["image"].convert('RGB')
            batch_images.append(image)
            
            if len(batch_images) >= config.batch_size:
                emb = encoder.encode(batch_images)
                embeddings.append(emb)
                pbar.update(len(batch_images))
                batch_images = []
                
                # Check if we hit a checkpoint
                current_n = sum(e.shape[0] for e in embeddings)
                while checkpoint_idx < len(config.elbow_checkpoints) and current_n >= config.elbow_checkpoints[checkpoint_idx]:
                    checkpoint_n = config.elbow_checkpoints[checkpoint_idx]
                    metrics = compute_elbow_metrics(torch.cat(embeddings)[:checkpoint_n])
                    results["checkpoints"].append(checkpoint_n)
                    results["metrics"].append(metrics)
                    print(f"\n  Checkpoint {checkpoint_n}: mean_knn={metrics['mean_knn_dist']:.4f}, coverage={metrics['coverage']:.4f}")
                    checkpoint_idx += 1
                
        except Exception:
            continue
    
    pbar.close()
    
    # Process remaining
    if batch_images:
        emb = encoder.encode(batch_images)
        embeddings.append(emb)
    
    # Final checkpoint if needed
    all_emb = torch.cat(embeddings)
    if checkpoint_idx < len(config.elbow_checkpoints):
        for cp in config.elbow_checkpoints[checkpoint_idx:]:
            if cp <= len(all_emb):
                metrics = compute_elbow_metrics(all_emb[:cp])
                results["checkpoints"].append(cp)
                results["metrics"].append(metrics)
    
    # Save and plot
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config.output_dir / "elbow_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_elbow(results, config.output_dir / "elbow_analysis.png")
    
    # Find elbow
    knn_dists = [m["mean_knn_dist"] for m in results["metrics"]]
    elbow_idx = find_elbow_point(knn_dists)
    recommended = results["checkpoints"][elbow_idx] if elbow_idx else results["checkpoints"][-1]
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: Use {recommended} samples as reference size")
    print(f"{'='*70}")
    
    return results


def compute_elbow_metrics(embeddings: torch.Tensor, k: int = 10, sample_n: int = 1000) -> Dict:
    """Compute metrics for elbow analysis."""
    emb_np = embeddings.numpy()
    n = len(emb_np)
    
    # Sample for efficiency
    if n > sample_n:
        idx = np.random.choice(n, sample_n, replace=False)
        sample = emb_np[idx]
    else:
        sample = emb_np
    
    # Pairwise cosine distances (sample)
    from scipy.spatial.distance import cdist
    dists = cdist(sample[:200], sample[:200], metric='cosine')
    triu_idx = np.triu_indices(len(dists), k=1)
    pairwise = dists[triu_idx]
    
    # k-NN distances
    from scipy.spatial import cKDTree
    tree = cKDTree(emb_np)
    knn_dists, _ = tree.query(sample, k=k+1)  # +1 because first is self
    knn_dists = knn_dists[:, 1:]  # Remove self
    
    return {
        "mean_pairwise_dist": float(np.mean(pairwise)),
        "std_pairwise_dist": float(np.std(pairwise)),
        "mean_knn_dist": float(np.mean(knn_dists)),
        "coverage": float(np.std(emb_np, axis=0).mean()),
    }


def find_elbow_point(values: List[float]) -> int:
    """Find elbow using second derivative."""
    if len(values) < 3:
        return len(values) - 1
    
    # Compute second derivative
    d2 = np.diff(np.diff(values))
    # Elbow is where second derivative is maximum (curve bends most)
    return int(np.argmax(np.abs(d2))) + 1


def plot_elbow(results: Dict, output_path: Path):
    """Plot elbow analysis results."""
    checkpoints = results["checkpoints"]
    metrics = results["metrics"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # k-NN distance (main elbow metric)
    ax = axes[0]
    knn = [m["mean_knn_dist"] for m in metrics]
    ax.plot(checkpoints, knn, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Reference Size")
    ax.set_ylabel("Mean k-NN Distance")
    ax.set_title("k-NN Distance (Lower = Denser)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Coverage
    ax = axes[1]
    coverage = [m["coverage"] for m in metrics]
    ax.plot(checkpoints, coverage, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel("Reference Size")
    ax.set_ylabel("Coverage (Mean Std)")
    ax.set_title("Embedding Space Coverage")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Pairwise distance
    ax = axes[2]
    pairwise = [m["mean_pairwise_dist"] for m in metrics]
    ax.plot(checkpoints, pairwise, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel("Reference Size")
    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title("Diversity (Pairwise Distance)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved elbow plot to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build WikiArt Reference Cloud")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="Number of samples for reference")
    parser.add_argument("--embedding-model", type=str, default="dino",
                        choices=["dino", "clip"],
                        help="Embedding model to use (dino or clip)")
    parser.add_argument("--cache-images", action="store_true",
                        help="Cache images locally for visualization")
    parser.add_argument("--elbow-analysis", action="store_true",
                        help="Run elbow analysis to find optimal size")
    parser.add_argument("--output-dir", type=str, 
                        default="/project/jevans/avi/wikiart_reference",
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str,
                        default="/project/jevans/avi/wikiart_cache",
                        help="Image cache directory")
    
    args = parser.parse_args()
    
    config = Config(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        num_samples=args.num_samples,
        embedding_model=args.embedding_model,
    )
    
    if args.elbow_analysis:
        elbow_analysis(config)
    else:
        build_wikiart_reference(config, cache_images=args.cache_images)


if __name__ == "__main__":
    main()
