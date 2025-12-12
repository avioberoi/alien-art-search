"""
Find K-Nearest WikiArt Images for Alien Art
=============================================
Given generated "alien" images, finds the K nearest images in WikiArt reference.
This helps validate that:
  - Low-scoring images are similar to WikiArt (expected)
  - High-scoring images are genuinely distant (alien)

Usage:
  python find_nearest_wikiart.py --search-dir outputs/paper_mapelites_xxx --k 5
"""

import torch
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


def load_reference(ref_path: Path) -> Tuple[torch.Tensor, List[Dict]]:
    """Load WikiArt reference with metadata."""
    print(f"Loading reference from {ref_path}...")
    with open(ref_path, 'rb') as f:
        ref = pickle.load(f)
    
    embeddings = ref["embeddings"]
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings
    
    metadata = ref.get("metadata", [{"idx": i} for i in range(len(embeddings))])
    
    print(f"  Loaded {len(embeddings)} reference embeddings")
    return embeddings, metadata


def load_search_embeddings(search_dir: Path) -> Tuple[torch.Tensor, List[Dict]]:
    """Load embeddings from search results."""
    # Try different log formats
    for log_name in ["search_log.json", "cma_log.json", "archive_data.json"]:
        log_path = search_dir / log_name
        if log_path.exists():
            break
    else:
        raise FileNotFoundError(f"No search log found in {search_dir}")
    
    print(f"Loading search results from {log_path}...")
    with open(log_path) as f:
        data = json.load(f)
    
    results = data.get("results", data.get("elites", []))
    
    # Load embeddings
    emb_path = search_dir / "embeddings.npy"
    if emb_path.exists():
        embeddings = torch.from_numpy(np.load(emb_path))
    else:
        # Need to recompute embeddings from images
        print("  Embeddings not found, will recompute...")
        embeddings = None
    
    metadata = []
    for i, r in enumerate(results):
        metadata.append({
            "idx": i,
            "novelty": r.get("novelty", 0),
            "image_path": r.get("image_path", ""),
            "prompt": r.get("prompt", ""),
        })
    
    return embeddings, metadata


def find_k_nearest(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    k: int = 5,
    faiss_index_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest reference embeddings for each query.
    
    Args:
        query_embeddings: Query embeddings
        reference_embeddings: Reference embeddings (used if no FAISS index)
        k: Number of neighbors
        faiss_index_path: Optional path to FAISS index for acceleration
    
    Returns:
        indices: [N, k] indices of nearest references
        distances: [N, k] cosine distances (1 - similarity)
    """
    query_np = query_embeddings.numpy() if isinstance(query_embeddings, torch.Tensor) else query_embeddings
    
    # Try FAISS first
    if faiss_index_path and Path(faiss_index_path).exists():
        try:
            import faiss
            print(f"  Using FAISS index: {faiss_index_path}")
            index = faiss.read_index(faiss_index_path)
            
            # FAISS with inner product = cosine similarity for normalized vectors
            # Returns similarity scores, convert to distance
            similarities, indices = index.search(query_np.astype(np.float32), k)
            distances = 1.0 - similarities  # cosine distance = 1 - similarity
            
            return indices, distances
        except ImportError:
            print("  FAISS not available, falling back to brute force")
        except Exception as e:
            print(f"  FAISS error: {e}, falling back to brute force")
    
    # Fallback: brute force
    ref_np = reference_embeddings.numpy() if isinstance(reference_embeddings, torch.Tensor) else reference_embeddings
    
    all_indices = []
    all_distances = []
    
    for i in tqdm(range(len(query_np)), desc="Finding nearest"):
        similarities = ref_np @ query_np[i]
        distances = 1.0 - similarities
        
        # Get k smallest
        top_k_idx = np.argpartition(distances, k)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
        
        all_indices.append(top_k_idx)
        all_distances.append(distances[top_k_idx])
    
    return np.array(all_indices), np.array(all_distances)


def create_visualization(
    search_metadata: List[Dict],
    ref_metadata: List[Dict],
    nearest_indices: np.ndarray,
    nearest_distances: np.ndarray,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    n_examples: int = 10,
    k: int = 5,
):
    """Create visualization showing alien images with their nearest WikiArt neighbors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by novelty (high to low)
    sorted_idx = sorted(range(len(search_metadata)), 
                       key=lambda i: search_metadata[i].get("novelty", 0), 
                       reverse=True)
    
    # Select examples: top N most alien, bottom N least alien
    top_alien = sorted_idx[:n_examples]
    bottom_alien = sorted_idx[-n_examples:]
    
    def create_comparison_figure(indices: List[int], title: str, filename: str):
        """Create figure comparing generated images with WikiArt neighbors."""
        n_rows = len(indices)
        n_cols = k + 1  # 1 for generated + k neighbors
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for row, idx in enumerate(indices):
            meta = search_metadata[idx]
            
            # Generated image
            ax = axes[row, 0] if n_rows > 1 else axes[0]
            img_path = Path(meta.get("image_path", ""))
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(f"Generated\nNovelty: {meta.get('novelty', 0):.3f}", fontsize=9)
            ax.axis('off')
            
            # K nearest WikiArt
            for col in range(k):
                ax = axes[row, col + 1] if n_rows > 1 else axes[col + 1]
                
                ref_idx = nearest_indices[idx, col]
                dist = nearest_distances[idx, col]
                ref_meta = ref_metadata[ref_idx] if ref_idx < len(ref_metadata) else {"idx": ref_idx}
                
                # Try to load WikiArt image
                if cache_dir and "image_path" in ref_meta:
                    ref_img_path = Path(ref_meta["image_path"])
                    if ref_img_path.exists():
                        img = Image.open(ref_img_path)
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, f"WikiArt #{ref_idx}", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f"WikiArt #{ref_idx}", ha='center', va='center', transform=ax.transAxes)
                
                # Safely get style (might not be present or might be wrong type)
                style_val = ref_meta.get("style", "unknown") if isinstance(ref_meta, dict) else "unknown"
                style = str(style_val)[:20] if style_val else "unknown"
                ax.set_title(f"Dist: {dist:.3f}\n{style}", fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Create visualizations
    create_comparison_figure(top_alien, "Most Alien Images vs Nearest WikiArt", "top_alien_vs_wikiart.png")
    create_comparison_figure(bottom_alien, "Least Alien Images vs Nearest WikiArt", "bottom_alien_vs_wikiart.png")
    
    # Helper to safely get style
    def get_style(ref_idx):
        if ref_idx < len(ref_metadata):
            meta = ref_metadata[ref_idx]
            if isinstance(meta, dict):
                return str(meta.get("style", "unknown"))
        return "unknown"
    
    # Save detailed results
    results = {
        "top_alien": [
            {
                "idx": idx,
                "novelty": search_metadata[idx].get("novelty", 0),
                "image_path": search_metadata[idx].get("image_path", ""),
                "nearest_wikiart": [
                    {
                        "ref_idx": int(nearest_indices[idx, j]),
                        "distance": float(nearest_distances[idx, j]),
                        "style": get_style(nearest_indices[idx, j]),
                    }
                    for j in range(k)
                ]
            }
            for idx in top_alien
        ],
        "bottom_alien": [
            {
                "idx": idx,
                "novelty": search_metadata[idx].get("novelty", 0),
                "image_path": search_metadata[idx].get("image_path", ""),
                "nearest_wikiart": [
                    {
                        "ref_idx": int(nearest_indices[idx, j]),
                        "distance": float(nearest_distances[idx, j]),
                        "style": get_style(nearest_indices[idx, j]),
                    }
                    for j in range(k)
                ]
            }
            for idx in bottom_alien
        ]
    }
    
    with open(output_dir / "nearest_wikiart_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: nearest_wikiart_analysis.json")


def main():
    parser = argparse.ArgumentParser(description="Find K-Nearest WikiArt Images")
    parser.add_argument("--search-dir", type=str, required=True,
                        help="Path to search output directory")
    parser.add_argument("--reference", type=str, 
                        default="/project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl",
                        help="Path to WikiArt reference")
    parser.add_argument("--faiss-index", type=str,
                        default="/project/jevans/avi/wikiart_reference/wikiart_dino_81444.faiss",
                        help="Path to FAISS index (optional, for faster search)")
    parser.add_argument("--cache-dir", type=str,
                        default="/project/jevans/avi/wikiart_cache",
                        help="Path to cached WikiArt images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: search-dir/wikiart_analysis)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbors")
    parser.add_argument("--n-examples", type=int, default=10,
                        help="Number of examples for visualization")
    
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    output_dir = Path(args.output_dir) if args.output_dir else search_dir / "wikiart_analysis"
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    faiss_index = args.faiss_index if hasattr(args, 'faiss_index') else None
    
    # Load data
    ref_embeddings, ref_metadata = load_reference(Path(args.reference))
    search_embeddings, search_metadata = load_search_embeddings(search_dir)
    
    if search_embeddings is None:
        print("ERROR: Could not load search embeddings")
        return
    
    # Find nearest (with FAISS acceleration if available)
    print(f"\nFinding {args.k} nearest WikiArt images...")
    nearest_idx, nearest_dist = find_k_nearest(
        search_embeddings, ref_embeddings, k=args.k, faiss_index_path=faiss_index
    )
    
    # Visualize
    print(f"\nCreating visualizations...")
    create_visualization(
        search_metadata, ref_metadata,
        nearest_idx, nearest_dist,
        output_dir, cache_dir,
        n_examples=args.n_examples, k=args.k
    )
    
    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
