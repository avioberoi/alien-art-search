"""
Low-Density Illumination: Measuring Distance from Reference Distributions
=========================================================================
This module provides metrics for measuring how "off-manifold" or "low-density"
an image is relative to reference distributions like:
1. WikiArt (real art)
2. Typical SD outputs (common generations)
3. ImageNet (natural images)

The key insight: novelty vs history measures open-endedness,
but distance from reference measures actual low-density illumination.

Usage:
  # Build reference embedding cloud
  python illumination.py build --source wikiart --path /path/to/wikiart --output ref_art.pkl
  python illumination.py build --source sd_typical --num_samples 1000 --output ref_sd.pkl
  
  # Analyze existing search results
  python illumination.py analyze --search_dir outputs/map_elites --reference ref_art.pkl
"""

import torch
import numpy as np
import pandas as pd
import re
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pickle
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# REFERENCE CLOUD
# =============================================================================

@dataclass
class ReferenceCloud:
    """A cloud of embeddings from a reference distribution."""
    embeddings: torch.Tensor  # [N, D]
    source: str  # "wikiart", "sd_typical", "imagenet", etc.
    embedding_type: str  # "dino" or "clip"
    num_samples: int
    metadata: Optional[dict] = None
    
    def save(self, path: Path):
        """Save to disk."""
        data = {
            "embeddings": self.embeddings.cpu(),
            "source": self.source,
            "embedding_type": self.embedding_type,
            "num_samples": self.num_samples,
            "metadata": self.metadata,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "ReferenceCloud":
        """Load from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            embeddings=data["embeddings"],
            source=data["source"],
            embedding_type=data["embedding_type"],
            num_samples=data["num_samples"],
            metadata=data.get("metadata"),
        )


def compute_density_score(embedding: torch.Tensor, reference: ReferenceCloud, k: int = 10) -> float:
    """
    Compute density score based on k-nearest neighbors in reference cloud.
    
    Lower score = further from reference = more "off-manifold"
    
    Returns: average cosine similarity to k nearest neighbors
    """
    # Compute similarities to all reference embeddings
    similarities = reference.embeddings @ embedding  # [N]
    
    # Get top-k
    topk_sims, _ = torch.topk(similarities, min(k, len(similarities)))
    
    # Average similarity to k nearest neighbors
    return topk_sims.mean().item()


def compute_novelty_from_reference(embedding: torch.Tensor, reference: ReferenceCloud) -> float:
    """
    Compute novelty as 1 - max(similarity to reference).
    
    This is analogous to our history-based novelty, but against a fixed reference.
    Higher = further from reference distribution.
    """
    similarities = reference.embeddings @ embedding
    return 1.0 - similarities.max().item()


def compute_percentile_novelty(embedding: torch.Tensor, reference: ReferenceCloud) -> float:
    """
    Compute what percentile of the reference distribution this embedding is at.
    
    Uses the distance to the closest reference point.
    Higher percentile = more unusual relative to reference.
    """
    similarities = reference.embeddings @ embedding
    max_sim = similarities.max().item()
    
    # Compute the distribution of max similarities within reference
    # (i.e., for each reference point, what's its max similarity to others?)
    # This is expensive, so we approximate with a sample
    
    # For now, just return the novelty score
    # A more sophisticated version would compare against the internal distribution
    return 1.0 - max_sim


# =============================================================================
# BUILD REFERENCE CLOUDS
# =============================================================================

def build_wikiart_cloud_from_hf(
    output_path: Path,
    embedding_type: str = "dino",
    sample_size: int = 5000,
    device: str = "cuda",
) -> ReferenceCloud:
    """
    Build reference cloud from HuggingFace WikiArt dataset (huggan/wikiart).
    
    This streams the dataset so no full download is required.
    Dataset contains 81,444 artworks with artist, genre, and style labels.
    """
    from datasets import load_dataset
    
    print(f"Building WikiArt reference cloud from HuggingFace ({embedding_type})...")
    print(f"  Loading dataset: huggan/wikiart")
    
    # Load dataset in streaming mode for efficiency
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    # Load embedding model
    if embedding_type == "dino":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    else:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
    model.eval()
    
    # Sample and embed images
    embeddings = []
    styles_seen = set()
    
    print(f"  Sampling and embedding {sample_size} images...")
    
    for i, sample in enumerate(tqdm(ds, total=sample_size, desc="Embedding WikiArt")):
        if i >= sample_size:
            break
            
        try:
            image = sample["image"].convert('RGB')
            styles_seen.add(sample.get("style", "unknown"))
            
            if embedding_type == "dino":
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            else:
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(image_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            
            embeddings.append(emb.squeeze(0).cpu())
            
        except Exception as e:
            continue
    
    print(f"  Successfully embedded {len(embeddings)} images")
    print(f"  Styles represented: {len(styles_seen)}")
    
    # Create cloud
    cloud = ReferenceCloud(
        embeddings=torch.stack(embeddings),
        source="wikiart_hf",
        embedding_type=embedding_type,
        num_samples=len(embeddings),
        metadata={
            "dataset": "huggan/wikiart",
            "styles_seen": list(styles_seen),
        },
    )
    
    cloud.save(output_path)
    print(f"  Saved to {output_path}")
    
    return cloud

# =============================================================================
# ANALYZE SEARCH RESULTS
# =============================================================================

def analyze_with_reference(
    search_dir: Path,
    reference_path: Path,
    output_dir: Path,
    embedding_type: str = "dino",
    device: str = "cuda",
):
    """
    Analyze existing search results against a reference cloud.
    Adds "distance from reference" as a second axis of analysis.
    """
    print("=" * 70)
    print("ILLUMINATION ANALYSIS")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load reference
    print(f"\nLoading reference cloud from {reference_path}...")
    reference = ReferenceCloud.load(reference_path)
    reference.embeddings = reference.embeddings.to(device)
    print(f"  Source: {reference.source}")
    print(f"  Embedding type: {reference.embedding_type}")
    print(f"  Samples: {reference.num_samples}")
    
    # Check embedding type compatibility
    if reference.embedding_type != embedding_type:
        print(f"\n  WARNING: Reference uses {reference.embedding_type} but analysis uses {embedding_type}")
        print(f"  Switching to {reference.embedding_type} for consistency.")
        embedding_type = reference.embedding_type
    
    # Load search results
    archive_path = search_dir / "archive_data.json"
    if not archive_path.exists():
        # Try search_log.json for random/CMA search
        archive_path = search_dir / "search_log.json"
        if not archive_path.exists():
            # Try cma_log.json
            archive_path = search_dir / "cma_log.json"
    
    with open(archive_path) as f:
        search_data = json.load(f)
    
    results = search_data.get("results", search_data.get("elites", []))
    print(f"  Loaded {len(results)} results from {archive_path}")
    
    # Load embedding model
    if embedding_type == "dino":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    else:
        import open_clip
        model, _, processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
    model.eval()
    
    # Analyze each result
    print("\nComputing reference distances...")
    
    novelties = []  # From search history
    ref_distances = []  # From reference cloud
    
    for r in tqdm(results, desc="Analyzing"):
        # Get novelty from search
        novelty = r.get("novelty", 0)
        novelties.append(novelty)
        
        # Load and embed image
        img_path = Path(r["image_path"])
        if not img_path.exists():
            ref_distances.append(0)
            continue
        
        image = Image.open(img_path).convert('RGB')
        
        if embedding_type == "dino":
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = emb / emb.norm(dim=-1, keepdim=True)
        else:
            image_tensor = processor(image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
        
        emb = emb.squeeze(0)
        
        # Compute distance from reference
        ref_dist = compute_novelty_from_reference(emb, reference)
        ref_distances.append(ref_dist)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    create_illumination_plots(novelties, ref_distances, reference.source, output_dir)
    
    # Save analysis
    analysis = {
        "search_dir": str(search_dir),
        "reference_source": str(reference.source),
        "reference_embedding_type": str(reference.embedding_type),
        "stats": {
            "novelty_mean": float(np.mean(novelties)),
            "novelty_std": float(np.std(novelties)),
            "ref_distance_mean": float(np.mean(ref_distances)),
            "ref_distance_std": float(np.std(ref_distances)),
            "correlation": float(np.corrcoef(novelties, ref_distances)[0, 1]) if len(novelties) > 1 else 0.0,
            "combined_score_mean": float(np.mean(np.array(novelties) * np.array(ref_distances))),
        },
        "results": [
            {"novelty": float(n), "ref_distance": float(d)}
            for n, d in zip(novelties, ref_distances)
        ]
    }
    
    with open(output_dir / "illumination_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"\nCorrelation (novelty vs ref_distance): {analysis['stats']['correlation']:.4f}")
    
    return novelties, ref_distances


def create_illumination_plots(novelties: List[float], ref_distances: List[float], 
                               ref_source: str, output_dir: Path):
    """Create illumination analysis visualizations."""
    
    # 1. 2D scatter: novelty vs reference distance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1 = axes[0]
    scatter = ax1.scatter(novelties, ref_distances, 
                          c=np.arange(len(novelties)), cmap='viridis',
                          alpha=0.7, s=30)
    ax1.set_xlabel('Novelty (vs search history)')
    ax1.set_ylabel(f'Distance from {ref_source}')
    ax1.set_title('Illumination: Novelty vs Reference Distance')
    ax1.grid(True, alpha=0.3)
    
    # Add quadrant labels
    mid_nov = np.median(novelties)
    mid_dist = np.median(ref_distances)
    ax1.axhline(mid_dist, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(mid_nov, color='red', linestyle='--', alpha=0.5)
    
    # Quadrant annotations
    ax1.text(0.02, 0.98, "Repeated\n(low both)", transform=ax1.transAxes, 
             fontsize=9, va='top', alpha=0.7)
    ax1.text(0.98, 0.98, "TRULY ALIEN\n(high both)", transform=ax1.transAxes, 
             fontsize=9, va='top', ha='right', fontweight='bold', color='green')
    ax1.text(0.02, 0.02, "Reference-like\n(typical)", transform=ax1.transAxes, 
             fontsize=9, va='bottom', alpha=0.7)
    ax1.text(0.98, 0.02, "Novel but familiar", transform=ax1.transAxes, 
             fontsize=9, va='bottom', ha='right', alpha=0.7)
    
    plt.colorbar(scatter, ax=ax1, label='Iteration')
    
    # 2. Distributions
    ax2 = axes[1]
    ax2.hist(novelties, bins=25, alpha=0.5, label='Novelty (history)')
    ax2.hist(ref_distances, bins=25, alpha=0.5, label=f'Distance ({ref_source})')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distributions')
    ax2.legend()
    
    # 3. Combined score
    combined = np.array(novelties) * np.array(ref_distances)
    ax3 = axes[2]
    ax3.hist(combined, bins=25, edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(combined), color='r', linestyle='--', 
                label=f'Mean: {np.mean(combined):.4f}')
    ax3.set_xlabel('Combined Score (novelty × ref_distance)')
    ax3.set_ylabel('Count')
    ax3.set_title('True Alien Score')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_analysis.png", dpi=150)
    plt.close()
    
    print(f"  Saved: illumination_analysis.png")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Low-Density Illumination Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build reference cloud")
    build_parser.add_argument("--source", type=str, required=True,
                              choices=["wikiart", "wikiart_csv", "wikiart_hf", "sd_typical"],
                              help="Source for reference distribution")
    build_parser.add_argument("--path", type=str,
                              help="Path to source data (for wikiart/wikiart_csv)")
    build_parser.add_argument("--num-samples", type=int, default=10000,
                              help="Number of samples")
    build_parser.add_argument("--output", type=str, required=True,
                              help="Output path for reference cloud")
    build_parser.add_argument("--embedding", type=str, default="dino",
                              choices=["dino", "clip"],
                              help="Embedding type (ignored for wikiart_csv which uses CLIP)")
    build_parser.add_argument("--feature-column", type=str, default="Feature",
                              help="Column name containing embeddings in CSV")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze search results")
    analyze_parser.add_argument("--search-dir", type=str, required=True,
                                help="Path to search output directory")
    analyze_parser.add_argument("--reference", type=str, required=True,
                                help="Path to reference cloud")
    analyze_parser.add_argument("--output-dir", type=str, default="outputs/illumination",
                                help="Output directory")
    analyze_parser.add_argument("--embedding", type=str, default="dino",
                                choices=["dino", "clip"],
                                help="Embedding type")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.command == "build":
        if args.source == "wikiart":
            if not args.path:
                print("Error: --path required for wikiart source")
                return
            build_wikiart_cloud(
                Path(args.path),
                Path(args.output),
                embedding_type=args.embedding,
                sample_size=args.num_samples,
                device=device,
            )
        elif args.source == "wikiart_csv":
            if not args.path:
                print("Error: --path required for wikiart_csv source (path to CSV file)")
                return
            build_wikiart_cloud_from_csv(
                Path(args.path),
                Path(args.output),
                sample_size=args.num_samples,
                feature_column=args.feature_column,
            )
        elif args.source == "wikiart_hf":
            build_wikiart_cloud_from_hf(
                Path(args.output),
                embedding_type=args.embedding,
                sample_size=args.num_samples,
                device=device,
            )
        elif args.source == "sd_typical":
            build_sd_typical_cloud(
                Path(args.output),
                embedding_type=args.embedding,
                num_samples=args.num_samples,
                device=device,
            )
    
    elif args.command == "analyze":
        analyze_with_reference(
            Path(args.search_dir),
            Path(args.reference),
            Path(args.output_dir),
            embedding_type=args.embedding,
            device=device,
        )


if __name__ == "__main__":
    main()