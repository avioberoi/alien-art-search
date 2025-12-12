"""
MAP-Elites Analysis: Evolution, Illumination, and Off-Manifold
=============================================================================
This script answers the key questions:
1. How do elites evolve? (Elite Evolution Tracking)
2. Are we reaching low-density regions? (Illumination Story)
3. Are we going off-manifold? (Geometric Analysis)

Imports functionality from:
- geometric_analysis.py: Off-manifold metrics
- find_nearest_wikiart.py: WikiArt neighbor analysis
- illumination.py: Density-based metrics

Usage:
  python analyze_mapelites.py --search-dir outputs/paper_mapelites_xxx \
      --reference /project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl \
      --output-dir outputs/analysis_xxx
"""

import torch
import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import stats

# Import from our modules
from geometric_analysis import (
    GeometricConfig,
    GeometricReferenceModel,
    load_embeddings_from_file,
)


# =============================================================================
# ELITE EVOLUTION TRACKING
# =============================================================================

@dataclass
class EliteSnapshot:
    """Snapshot of an elite at a specific iteration."""
    iteration: int
    prompt: str
    novelty: float
    image_path: str
    replaced_at: Optional[int] = None  # When this elite was replaced


def track_elite_evolution(search_dir: Path) -> Dict[Tuple[int, int], List[EliteSnapshot]]:
    """
    Track how each cell's elite evolved over time.
    
    Note: Standard MAP-Elites only saves final elites. 
    This reconstructs evolution from iteration numbers.
    
    Returns:
        Dict mapping cell -> list of snapshots (chronological)
    """
    archive_path = search_dir / "archive_data.json"
    if not archive_path.exists():
        raise FileNotFoundError(f"No archive_data.json found in {search_dir}")
    
    with open(archive_path) as f:
        data = json.load(f)
    
    elites = data.get("elites", [])
    
    # Group by cell
    cell_history: Dict[Tuple[int, int], List[EliteSnapshot]] = {}
    
    for elite in elites:
        cell = tuple(elite["cell"])
        snapshot = EliteSnapshot(
            iteration=elite["iteration"],
            prompt=elite["prompt"],
            novelty=elite["novelty"],
            image_path=elite["image_path"],
        )
        
        if cell not in cell_history:
            cell_history[cell] = []
        cell_history[cell].append(snapshot)
    
    # Sort by iteration (most recent is the final elite)
    for cell in cell_history:
        cell_history[cell].sort(key=lambda s: s.iteration)
    
    return cell_history


def analyze_elite_turnover(cell_history: Dict[Tuple[int, int], List[EliteSnapshot]]) -> Dict:
    """
    Analyze how often elites are replaced.
    
    Returns statistics about elite turnover.
    """
    # Note: We only have final elites, so we estimate from iteration spread
    iterations = []
    novelties = []
    
    for cell, snapshots in cell_history.items():
        for s in snapshots:
            iterations.append(s.iteration)
            novelties.append(s.novelty)
    
    # When were most elites discovered?
    iteration_distribution = {
        'early_quartile': np.percentile(iterations, 25),
        'median': np.percentile(iterations, 50),
        'late_quartile': np.percentile(iterations, 75),
        'max': max(iterations),
    }
    
    # Correlation between iteration and novelty
    # (Do later elites have higher novelty?)
    correlation = np.corrcoef(iterations, novelties)[0, 1] if len(iterations) > 1 else 0.0
    
    return {
        'num_cells_filled': len(cell_history),
        'iteration_distribution': iteration_distribution,
        'novelty_iteration_correlation': float(correlation),
        'avg_novelty_early': float(np.mean([n for i, n in zip(iterations, novelties) 
                                            if i < iteration_distribution['median']])) if iterations else 0,
        'avg_novelty_late': float(np.mean([n for i, n in zip(iterations, novelties) 
                                           if i >= iteration_distribution['median']])) if iterations else 0,
    }


def create_evolution_visualization(
    cell_history: Dict[Tuple[int, int], List[EliteSnapshot]],
    grid_size: int,
    output_dir: Path,
):
    """
    Visualize when each cell was filled and with what novelty.
    """
    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. When was each cell filled (iteration)
    iteration_grid = np.full((grid_size, grid_size), np.nan)
    for cell, snapshots in cell_history.items():
        x, y = cell
        if 0 <= x < grid_size and 0 <= y < grid_size:
            iteration_grid[y, x] = snapshots[-1].iteration  # Final elite's iteration
    
    im1 = axes[0].imshow(iteration_grid, cmap='viridis', origin='lower')
    axes[0].set_title('When Cells Were Filled (Iteration)', fontsize=12)
    axes[0].set_xlabel('Cell X')
    axes[0].set_ylabel('Cell Y')
    plt.colorbar(im1, ax=axes[0], label='Iteration')
    
    # 2. Final novelty in each cell
    novelty_grid = np.full((grid_size, grid_size), np.nan)
    for cell, snapshots in cell_history.items():
        x, y = cell
        if 0 <= x < grid_size and 0 <= y < grid_size:
            novelty_grid[y, x] = snapshots[-1].novelty
    
    im2 = axes[1].imshow(novelty_grid, cmap='hot', origin='lower')
    axes[1].set_title('Final Novelty Score per Cell', fontsize=12)
    axes[1].set_xlabel('Cell X')
    axes[1].set_ylabel('Cell Y')
    plt.colorbar(im2, ax=axes[1], label='Novelty')
    
    # 3. Coverage over time
    all_iterations = sorted(set(
        s.iteration for snapshots in cell_history.values() for s in snapshots
    ))
    coverage_curve = []
    cells_filled = set()
    
    for it in all_iterations:
        for cell, snapshots in cell_history.items():
            for s in snapshots:
                if s.iteration <= it:
                    cells_filled.add(cell)
        coverage_curve.append(len(cells_filled) / (grid_size ** 2))
    
    axes[2].plot(all_iterations, coverage_curve, 'g-', linewidth=2)
    axes[2].fill_between(all_iterations, 0, coverage_curve, alpha=0.3, color='green')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Coverage')
    axes[2].set_title('Archive Coverage Over Time')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "elite_evolution.png", dpi=150)
    plt.close()
    print(f"  Saved: elite_evolution.png")


def create_elite_image_gallery(
    cell_history: Dict[Tuple[int, int], List[EliteSnapshot]],
    output_dir: Path,
    n_top: int = 16,
    n_bottom: int = 16,
):
    """
    Create image galleries showing actual elite images.
    
    Shows:
    1. Top N most novel elites with their prompts
    2. Bottom N least novel elites with their prompts
    3. Early vs Late elites comparison
    """
    # Flatten all elites
    all_elites = []
    for cell, snapshots in cell_history.items():
        for s in snapshots:
            all_elites.append((cell, s))
    
    # Sort by novelty
    sorted_by_novelty = sorted(all_elites, key=lambda x: x[1].novelty, reverse=True)
    
    # Sort by iteration
    sorted_by_iteration = sorted(all_elites, key=lambda x: x[1].iteration)
    
    def create_gallery(elites_list, title, filename, show_cell=True):
        """Create a gallery of elite images."""
        n = len(elites_list)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(elites_list):
                cell, elite = elites_list[idx]
                img_path = Path(elite.image_path)
                
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                
                # Title with key info
                cell_str = f"Cell {cell}" if show_cell else ""
                ax.set_title(
                    f"N={elite.novelty:.3f} | Iter {elite.iteration}\n"
                    f"{elite.prompt[:50]}...",
                    fontsize=8
                )
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")
    
    # 1. Top novel elites
    create_gallery(
        sorted_by_novelty[:n_top],
        f"Top {n_top} Most Novel Elites",
        "elite_top_novel_images.png"
    )
    
    # 2. Bottom novel elites
    create_gallery(
        sorted_by_novelty[-n_bottom:],
        f"Bottom {n_bottom} Least Novel Elites",
        "elite_bottom_novel_images.png"
    )
    
    # 3. Early elites (first 25% by iteration)
    n_early = len(sorted_by_iteration) // 4
    create_gallery(
        sorted_by_iteration[:min(16, n_early)],
        "Early Elites (First 25% of Iterations)",
        "elite_early_images.png"
    )
    
    # 4. Late elites (last 25% by iteration)
    create_gallery(
        sorted_by_iteration[-min(16, n_early):],
        "Late Elites (Last 25% of Iterations)",
        "elite_late_images.png"
    )
    
    # 5. Side-by-side: Early vs Late comparison
    n_compare = 8
    early_elites = sorted_by_iteration[:n_compare]
    late_elites = sorted_by_iteration[-n_compare:]
    
    fig, axes = plt.subplots(2, n_compare, figsize=(24, 8))
    fig.suptitle("Elite Evolution: Early vs Late Iterations", fontsize=14, fontweight='bold')
    
    # Early row
    for idx in range(n_compare):
        ax = axes[0, idx]
        if idx < len(early_elites):
            cell, elite = early_elites[idx]
            img_path = Path(elite.image_path)
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(f"Iter {elite.iteration}\nN={elite.novelty:.3f}", fontsize=9)
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel("EARLY", fontsize=12, fontweight='bold')
    
    # Late row
    for idx in range(n_compare):
        ax = axes[1, idx]
        if idx < len(late_elites):
            cell, elite = late_elites[idx]
            img_path = Path(elite.image_path)
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(f"Iter {elite.iteration}\nN={elite.novelty:.3f}", fontsize=9)
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel("LATE", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "elite_early_vs_late.png", dpi=150)
    plt.close()
    print(f"  Saved: elite_early_vs_late.png")


# =============================================================================
# ILLUMINATION ANALYSIS
# =============================================================================

def compute_illumination_metrics(
    embeddings: np.ndarray,
    novelties: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute illumination metrics: how far are we from the reference manifold?
    
    Args:
        embeddings: [N, D] generated image embeddings
        novelties: [N] novelty scores from search
        reference_embeddings: [M, D] WikiArt embeddings
        k: Number of neighbors for density estimation
    
    Returns:
        Dict of metric arrays
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Build k-NN index on reference
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(reference_embeddings)
    
    # Find distances to reference
    distances, indices = knn.kneighbors(embeddings)
    
    # Metrics
    avg_knn_distance = distances.mean(axis=1)
    min_distance = distances[:, 0]  # Distance to nearest reference
    
    # Cosine similarity to nearest
    nearest_similarity = 1.0 - min_distance
    
    # Local density (inverse of average k-NN distance)
    local_density = 1.0 / (avg_knn_distance + 1e-8)
    
    # Percentile rank (what % of reference is further than this?)
    # Compute pairwise distances within reference (sample for efficiency)
    sample_size = min(1000, len(reference_embeddings))
    ref_sample = reference_embeddings[np.random.choice(len(reference_embeddings), sample_size, replace=False)]
    ref_distances, _ = knn.kneighbors(ref_sample)
    ref_avg_distances = ref_distances.mean(axis=1)
    
    distance_percentile = np.array([
        stats.percentileofscore(ref_avg_distances, d) for d in avg_knn_distance
    ])
    
    return {
        'avg_knn_distance': avg_knn_distance,
        'min_distance': min_distance,
        'nearest_similarity': nearest_similarity,
        'local_density': local_density,
        'distance_percentile': distance_percentile,
        'novelty': novelties,
    }


def create_illumination_visualization(
    metrics: Dict[str, np.ndarray],
    output_dir: Path,
):
    """Create comprehensive illumination visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Novelty vs WikiArt Distance (The Key Plot)
    ax = axes[0, 0]
    scatter = ax.scatter(
        metrics['novelty'], 
        metrics['avg_knn_distance'],
        c=metrics['distance_percentile'],
        cmap='RdYlGn',
        alpha=0.6,
        s=30,
    )
    ax.set_xlabel('Novelty (vs Search History)', fontsize=11)
    ax.set_ylabel('Avg Distance to WikiArt', fontsize=11)
    ax.set_title('Illumination: Novelty vs Reference Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    med_nov = np.median(metrics['novelty'])
    med_dist = np.median(metrics['avg_knn_distance'])
    ax.axhline(med_dist, color='red', linestyle='--', alpha=0.5)
    ax.axvline(med_nov, color='red', linestyle='--', alpha=0.5)
    
    # Quadrant annotations
    ax.text(0.05, 0.95, "Repeated\n(low both)", transform=ax.transAxes, 
            fontsize=9, va='top', alpha=0.7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.95, 0.95, "TRULY ALIEN\n(high both)", transform=ax.transAxes, 
            fontsize=9, va='top', ha='right', fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.colorbar(scatter, ax=ax, label='Distance Percentile')
    
    # 2. Distribution of distances
    ax = axes[0, 1]
    ax.hist(metrics['avg_knn_distance'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(metrics['avg_knn_distance']), color='red', linestyle='--', 
               label=f'Median: {np.median(metrics["avg_knn_distance"]):.4f}')
    ax.set_xlabel('Avg k-NN Distance to WikiArt')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of WikiArt Distances')
    ax.legend()
    
    # 3. Distance percentile distribution
    ax = axes[0, 2]
    ax.hist(metrics['distance_percentile'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(50, color='red', linestyle='--', label='Median WikiArt')
    ax.set_xlabel('Distance Percentile (vs WikiArt)')
    ax.set_ylabel('Count')
    ax.set_title('How Unusual Are Our Images?')
    ax.legend()
    
    # Add interpretation
    pct_above_median = (metrics['distance_percentile'] > 50).mean() * 100
    ax.text(0.95, 0.95, f'{pct_above_median:.1f}% above\nWikiArt median', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Novelty vs Nearest Similarity
    ax = axes[1, 0]
    ax.scatter(metrics['novelty'], metrics['nearest_similarity'], alpha=0.5, s=20)
    ax.set_xlabel('Novelty (vs Search History)')
    ax.set_ylabel('Similarity to Nearest WikiArt')
    ax.set_title('Are Novel Images Similar to Any WikiArt?')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(metrics['novelty'], metrics['nearest_similarity'])[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Combined "Alien Score" 
    # Product of novelty and distance (both normalized)
    norm_novelty = (metrics['novelty'] - metrics['novelty'].min()) / (metrics['novelty'].max() - metrics['novelty'].min() + 1e-8)
    norm_distance = (metrics['avg_knn_distance'] - metrics['avg_knn_distance'].min()) / (metrics['avg_knn_distance'].max() - metrics['avg_knn_distance'].min() + 1e-8)
    combined_score = norm_novelty * norm_distance
    
    ax = axes[1, 1]
    ax.hist(combined_score, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(combined_score), color='red', linestyle='--', 
               label=f'Mean: {np.mean(combined_score):.4f}')
    ax.set_xlabel('Combined Alien Score (Novelty × Distance)')
    ax.set_ylabel('Count')
    ax.set_title('True Alien Score Distribution')
    ax.legend()
    
    # 6. Local density distribution
    ax = axes[1, 2]
    log_density = np.log10(metrics['local_density'] + 1e-8)
    ax.hist(log_density, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('log₁₀(Local Density)')
    ax.set_ylabel('Count')
    ax.set_title('Density in WikiArt Embedding Space')
    
    # Interpretation
    low_density_pct = (metrics['distance_percentile'] > 75).mean() * 100
    ax.text(0.95, 0.95, f'{low_density_pct:.1f}% in low-density\nregions (>75th pctl)', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: illumination_analysis.png")
    
    return combined_score


def create_nearest_neighbor_gallery(
    embeddings: np.ndarray,
    novelties: np.ndarray,
    image_paths: List[str],
    prompts: List[str],
    reference_embeddings: np.ndarray,
    ref_metadata: List[Dict],
    output_dir: Path,
    n_examples: int = 10,
    k: int = 5,
):
    """
    Create visual galleries showing elite images with their k-nearest WikiArt neighbors.
    
    Shows:
    1. Top N most novel elites with nearest WikiArt
    2. Bottom N least novel elites with nearest WikiArt
    """
    # Sort by novelty
    sorted_idx = np.argsort(novelties)[::-1]  # High to low
    top_idx = sorted_idx[:n_examples]
    bottom_idx = sorted_idx[-n_examples:]
    
    # Compute nearest neighbors
    print("  Computing nearest neighbors...")
    ref_np = reference_embeddings if isinstance(reference_embeddings, np.ndarray) else reference_embeddings.numpy()
    emb_np = embeddings if isinstance(embeddings, np.ndarray) else embeddings.numpy()
    
    # Find k nearest for all
    all_nn_idx = []
    all_nn_dist = []
    for i in range(len(emb_np)):
        similarities = ref_np @ emb_np[i]
        distances = 1.0 - similarities
        top_k = np.argpartition(distances, k)[:k]
        top_k = top_k[np.argsort(distances[top_k])]
        all_nn_idx.append(top_k)
        all_nn_dist.append(distances[top_k])
    
    all_nn_idx = np.array(all_nn_idx)
    all_nn_dist = np.array(all_nn_dist)
    
    def create_comparison_figure(indices, title, filename):
        """Create figure comparing elite images with WikiArt neighbors."""
        n_rows = len(indices)
        n_cols = k + 1  # 1 for elite + k neighbors
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(indices):
            # Elite image
            ax = axes[row, 0]
            img_path = Path(image_paths[idx])
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            prompt_short = prompts[idx][:40] + "..." if len(prompts[idx]) > 40 else prompts[idx]
            ax.set_title(f"Elite\nNovelty: {novelties[idx]:.3f}\n{prompt_short}", fontsize=8)
            ax.axis('off')
            
            # K nearest WikiArt
            for col in range(k):
                ax = axes[row, col + 1]
                ref_idx = all_nn_idx[idx, col]
                dist = all_nn_dist[idx, col]
                
                # Load WikiArt image - try image_path from metadata, fallback to constructing path from index
                ref_img_path = None
                if ref_idx < len(ref_metadata):
                    if 'image_path' in ref_metadata[ref_idx]:
                        ref_img_path = Path(ref_metadata[ref_idx]['image_path'])
                    else:
                        # Construct path from index (WikiArt cache naming convention)
                        ref_img_path = Path(f"/project/jevans/avi/wikiart_cache/{ref_idx:06d}.jpg")
                
                if ref_img_path and ref_img_path.exists():
                    img = Image.open(ref_img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, f"WikiArt #{ref_idx}", ha='center', va='center', transform=ax.transAxes)
                
                # Get style info
                style = "unknown"
                if ref_idx < len(ref_metadata):
                    style_val = ref_metadata[ref_idx].get('style', 'unknown')
                    style = str(style_val)[:15]
                
                ax.set_title(f"NN #{col+1}\nDist: {dist:.3f}\nStyle: {style}", fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Create galleries
    create_comparison_figure(top_idx, f"Top {n_examples} Most Novel Elites vs Nearest WikiArt", 
                            "novel_elites_vs_wikiart.png")
    create_comparison_figure(bottom_idx, f"Top {n_examples} Least Novel Elites vs Nearest WikiArt",
                            "typical_elites_vs_wikiart.png")
    
    # Save detailed analysis
    analysis = {
        "most_novel": [
            {
                "idx": int(idx),
                "novelty": float(novelties[idx]),
                "prompt": prompts[idx],
                "image_path": image_paths[idx],
                "nearest_wikiart": [
                    {
                        "ref_idx": int(all_nn_idx[idx, j]),
                        "distance": float(all_nn_dist[idx, j]),
                        "style": str(ref_metadata[all_nn_idx[idx, j]].get('style', 'unknown')) if all_nn_idx[idx, j] < len(ref_metadata) else "unknown"
                    }
                    for j in range(k)
                ]
            }
            for idx in top_idx
        ],
        "least_novel": [
            {
                "idx": int(idx),
                "novelty": float(novelties[idx]),
                "prompt": prompts[idx],
                "image_path": image_paths[idx],
                "nearest_wikiart": [
                    {
                        "ref_idx": int(all_nn_idx[idx, j]),
                        "distance": float(all_nn_dist[idx, j]),
                        "style": str(ref_metadata[all_nn_idx[idx, j]].get('style', 'unknown')) if all_nn_idx[idx, j] < len(ref_metadata) else "unknown"
                    }
                    for j in range(k)
                ]
            }
            for idx in bottom_idx
        ]
    }
    
    with open(output_dir / "nearest_neighbor_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: nearest_neighbor_analysis.json")


# =============================================================================
# OFF-MANIFOLD ANALYSIS
# =============================================================================

def run_offmanifold_analysis(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    output_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Run comprehensive off-manifold analysis using geometric metrics.
    """
    print("\nFitting geometric reference model...")
    config = GeometricConfig()
    model = GeometricReferenceModel(config)
    model.fit(reference_embeddings)
    
    print("\nComputing off-manifold metrics...")
    metrics = model.compute_all_metrics(embeddings)
    
    return metrics


def create_offmanifold_visualization(
    metrics: Dict[str, np.ndarray],
    novelties: np.ndarray,
    output_dir: Path,
):
    """Create off-manifold analysis visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mahalanobis Distance Distribution
    ax = axes[0, 0]
    ax.hist(metrics['mahalanobis'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(metrics['mahalanobis']), color='red', linestyle='--',
               label=f'Median: {np.median(metrics["mahalanobis"]):.2f}')
    ax.set_xlabel('Mahalanobis Distance')
    ax.set_ylabel('Count')
    ax.set_title('Mahalanobis Distance from WikiArt')
    ax.legend()
    
    # 2. Reconstruction Error
    ax = axes[0, 1]
    ax.hist(metrics['reconstruction_error'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.median(metrics['reconstruction_error']), color='red', linestyle='--',
               label=f'Median: {np.median(metrics["reconstruction_error"]):.4f}')
    ax.set_xlabel('PCA Reconstruction Error (SPE)')
    ax.set_ylabel('Count')
    ax.set_title('Distance from PCA Subspace')
    ax.legend()
    
    # 3. Low-Variance PC Magnitude
    ax = axes[0, 2]
    ax.hist(metrics['low_pc_magnitude'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.median(metrics['low_pc_magnitude']), color='red', linestyle='--',
               label=f'Median: {np.median(metrics["low_pc_magnitude"]):.4f}')
    ax.set_xlabel('Low-Variance PC Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Activity in Rare Directions')
    ax.legend()
    
    # 4. Novelty vs Mahalanobis
    ax = axes[1, 0]
    scatter = ax.scatter(novelties, metrics['mahalanobis'], 
                         c=metrics['mahalanobis_percentile'], cmap='RdYlGn',
                         alpha=0.6, s=30)
    ax.set_xlabel('Novelty (vs Search History)')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_title('Novelty vs Statistical Distance')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Mahalanobis Percentile')
    
    corr = np.corrcoef(novelties, metrics['mahalanobis'])[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Novelty vs k-NN Distance
    ax = axes[1, 1]
    ax.scatter(novelties, metrics['knn_distance'], alpha=0.5, s=20, color='orange')
    ax.set_xlabel('Novelty (vs Search History)')
    ax.set_ylabel('k-NN Distance')
    ax.set_title('Novelty vs Local Density')
    ax.grid(True, alpha=0.3)
    
    corr = np.corrcoef(novelties, metrics['knn_distance'])[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Combined Alien Score Distribution
    ax = axes[1, 2]
    ax.hist(metrics['combined_alien_score'], bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.axvline(np.mean(metrics['combined_alien_score']), color='blue', linestyle='--',
               label=f'Mean: {np.mean(metrics["combined_alien_score"]):.4f}')
    ax.set_xlabel('Combined Alien Score')
    ax.set_ylabel('Count')
    ax.set_title('Overall Off-Manifold Score')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "offmanifold_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: offmanifold_analysis.png")


# =============================================================================
# TOP ALIEN GALLERY
# =============================================================================

def create_top_alien_gallery(
    search_dir: Path,
    illumination_scores: np.ndarray,
    offmanifold_metrics: Dict[str, np.ndarray],
    output_dir: Path,
    n_images: int = 16,
):
    """
    Create gallery of top alien images based on combined scores.
    """
    # Load archive data
    with open(search_dir / "archive_data.json") as f:
        data = json.load(f)
    elites = data.get("elites", [])
    
    # Combined score: product of all normalized metrics
    norm_illumination = (illumination_scores - illumination_scores.min()) / (illumination_scores.max() - illumination_scores.min() + 1e-8)
    norm_mahal = (offmanifold_metrics['mahalanobis'] - offmanifold_metrics['mahalanobis'].min()) / (offmanifold_metrics['mahalanobis'].max() - offmanifold_metrics['mahalanobis'].min() + 1e-8)
    norm_knn = (offmanifold_metrics['knn_distance'] - offmanifold_metrics['knn_distance'].min()) / (offmanifold_metrics['knn_distance'].max() - offmanifold_metrics['knn_distance'].min() + 1e-8)
    
    combined_scores = (norm_illumination * norm_mahal * norm_knn) ** (1/3)  # Geometric mean
    
    # Get top indices
    top_indices = np.argsort(combined_scores)[-n_images:][::-1]
    
    # Create gallery
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle('Top Alien Images (Combined Score: Illumination × Mahalanobis × k-NN)', 
                 fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(top_indices):
            elite_idx = top_indices[idx]
            elite = elites[elite_idx]
            
            img_path = Path(elite['image_path'])
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            
            ax.set_title(
                f"Score: {combined_scores[elite_idx]:.3f}\n"
                f"N: {elite['novelty']:.3f} | M: {offmanifold_metrics['mahalanobis'][elite_idx]:.1f}",
                fontsize=9
            )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_alien_combined.png", dpi=150)
    plt.close()
    print(f"  Saved: top_alien_combined.png")
    
    return combined_scores, top_indices


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis(
    search_dir: Path,
    reference_path: Path,
    output_dir: Path,
):
    """Run complete MAP-Elites analysis."""
    print("=" * 70)
    print("COMPREHENSIVE MAP-ELITES ANALYSIS")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load search data
    with open(search_dir / "archive_data.json") as f:
        archive_data = json.load(f)
    
    config = archive_data.get("config", {})
    elites = archive_data.get("elites", [])
    grid_size = config.get("grid_size", 15)
    
    print(f"\nSearch Directory: {search_dir}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Elites: {len(elites)}")
    print(f"Coverage: {len(elites) / (grid_size ** 2) * 100:.1f}%")
    
    # Load embeddings
    emb_path = search_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found in {search_dir}")
    
    # Note: embeddings.npy contains ALL generated embeddings, not just elites
    # We need to extract elite embeddings based on their iteration indices
    all_embeddings = np.load(emb_path)
    print(f"  Loaded {len(all_embeddings)} total embeddings")
    
    # For MAP-Elites, we need to match elites to embeddings
    # The elite's iteration index corresponds to the embedding index
    elite_indices = [e['iteration'] for e in elites]
    
    # Handle case where indices exceed array size (shouldn't happen but be safe)
    valid_mask = np.array(elite_indices) < len(all_embeddings)
    if not valid_mask.all():
        print(f"  Warning: {(~valid_mask).sum()} elites have invalid indices")
        elite_indices = [i for i, v in zip(elite_indices, valid_mask) if v]
        elites = [e for e, v in zip(elites, valid_mask) if v]
    
    embeddings = all_embeddings[elite_indices]
    novelties = np.array([e['novelty'] for e in elites])
    print(f"  Using {len(embeddings)} elite embeddings")
    
    # Load reference
    print(f"\nLoading WikiArt reference from {reference_path}...")
    with open(reference_path, 'rb') as f:
        ref_data = pickle.load(f)
    reference_embeddings = ref_data['embeddings']
    if isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = reference_embeddings.numpy()
    print(f"  Reference: {len(reference_embeddings)} WikiArt embeddings")
    
    # ==========================================================================
    # 1. ELITE EVOLUTION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. ELITE EVOLUTION ANALYSIS")
    print("=" * 70)
    
    cell_history = track_elite_evolution(search_dir)
    turnover_stats = analyze_elite_turnover(cell_history)
    
    print(f"  Cells filled: {turnover_stats['num_cells_filled']}")
    print(f"  Iteration distribution:")
    print(f"    25th percentile: {turnover_stats['iteration_distribution']['early_quartile']:.0f}")
    print(f"    Median: {turnover_stats['iteration_distribution']['median']:.0f}")
    print(f"    75th percentile: {turnover_stats['iteration_distribution']['late_quartile']:.0f}")
    print(f"  Novelty-iteration correlation: {turnover_stats['novelty_iteration_correlation']:.3f}")
    print(f"  Avg novelty (early): {turnover_stats['avg_novelty_early']:.4f}")
    print(f"  Avg novelty (late): {turnover_stats['avg_novelty_late']:.4f}")
    
    create_evolution_visualization(cell_history, grid_size, output_dir)
    
    # Create actual image galleries
    print("\n  Creating elite image galleries...")
    create_elite_image_gallery(cell_history, output_dir, n_top=16, n_bottom=16)
    
    # ==========================================================================
    # 2. ILLUMINATION ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. ILLUMINATION ANALYSIS (Low-Density Regions)")
    print("=" * 70)
    
    illumination_metrics = compute_illumination_metrics(
        embeddings, novelties, reference_embeddings, k=10
    )
    
    print(f"  Avg k-NN distance: {illumination_metrics['avg_knn_distance'].mean():.4f}")
    print(f"  Median distance percentile: {np.median(illumination_metrics['distance_percentile']):.1f}")
    print(f"  % above 75th percentile: {(illumination_metrics['distance_percentile'] > 75).mean() * 100:.1f}%")
    print(f"  % above 90th percentile: {(illumination_metrics['distance_percentile'] > 90).mean() * 100:.1f}%")
    
    illumination_scores = create_illumination_visualization(illumination_metrics, output_dir)
    
    # Create nearest neighbor galleries (both high and low novelty)
    print("\n  Creating nearest neighbor galleries...")
    ref_metadata = ref_data.get('metadata', [{'idx': i} for i in range(len(reference_embeddings))])
    image_paths = [e.get('image_path', '') for e in elites]
    prompts = [e.get('prompt', '') for e in elites]
    create_nearest_neighbor_gallery(
        embeddings, novelties, image_paths, prompts,
        reference_embeddings, ref_metadata, output_dir,
        n_examples=10, k=5
    )
    
    # ==========================================================================
    # 3. OFF-MANIFOLD ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. OFF-MANIFOLD ANALYSIS (Geometric Metrics)")
    print("=" * 70)
    
    offmanifold_metrics = run_offmanifold_analysis(
        embeddings, reference_embeddings, output_dir
    )
    
    print(f"\nOff-manifold statistics:")
    print(f"  Mahalanobis distance: {offmanifold_metrics['mahalanobis'].mean():.2f} ± {offmanifold_metrics['mahalanobis'].std():.2f}")
    print(f"  Reconstruction error: {offmanifold_metrics['reconstruction_error'].mean():.4f} ± {offmanifold_metrics['reconstruction_error'].std():.4f}")
    print(f"  Low-PC magnitude: {offmanifold_metrics['low_pc_magnitude'].mean():.4f} ± {offmanifold_metrics['low_pc_magnitude'].std():.4f}")
    print(f"  k-NN distance: {offmanifold_metrics['knn_distance'].mean():.4f} ± {offmanifold_metrics['knn_distance'].std():.4f}")
    
    create_offmanifold_visualization(offmanifold_metrics, novelties, output_dir)
    
    # ==========================================================================
    # 4. TOP ALIEN GALLERY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. TOP ALIEN GALLERY")
    print("=" * 70)
    
    combined_scores, top_indices = create_top_alien_gallery(
        search_dir, illumination_scores, offmanifold_metrics, output_dir
    )
    
    # ==========================================================================
    # 5. SUMMARY STATISTICS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)
    
    # Key question: Are we reaching low-density / off-manifold regions?
    pct_low_density = (illumination_metrics['distance_percentile'] > 75).mean() * 100
    pct_high_mahal = (offmanifold_metrics['mahalanobis_percentile'] > 75).mean() * 100
    
    print(f"\n  KEY FINDINGS:")
    print(f"  • {pct_low_density:.1f}% of elites in low-density regions (>75th percentile distance)")
    print(f"  • {pct_high_mahal:.1f}% of elites have high Mahalanobis distance (>75th percentile)")
    print(f"  • Novelty-Mahalanobis correlation: {np.corrcoef(novelties, offmanifold_metrics['mahalanobis'])[0,1]:.3f}")
    print(f"  • Later elites {'have higher' if turnover_stats['avg_novelty_late'] > turnover_stats['avg_novelty_early'] else 'have similar'} novelty than early ones")
    
    # Save complete analysis
    analysis_results = {
        'config': config,
        'evolution': turnover_stats,
        'illumination': {
            'avg_knn_distance_mean': float(illumination_metrics['avg_knn_distance'].mean()),
            'avg_knn_distance_std': float(illumination_metrics['avg_knn_distance'].std()),
            'median_distance_percentile': float(np.median(illumination_metrics['distance_percentile'])),
            'pct_above_75': float((illumination_metrics['distance_percentile'] > 75).mean() * 100),
            'pct_above_90': float((illumination_metrics['distance_percentile'] > 90).mean() * 100),
        },
        'offmanifold': {
            'mahalanobis_mean': float(offmanifold_metrics['mahalanobis'].mean()),
            'mahalanobis_std': float(offmanifold_metrics['mahalanobis'].std()),
            'reconstruction_error_mean': float(offmanifold_metrics['reconstruction_error'].mean()),
            'knn_distance_mean': float(offmanifold_metrics['knn_distance'].mean()),
            'pct_high_mahal': float(pct_high_mahal),
        },
        'correlations': {
            'novelty_vs_mahalanobis': float(np.corrcoef(novelties, offmanifold_metrics['mahalanobis'])[0,1]),
            'novelty_vs_knn_distance': float(np.corrcoef(novelties, offmanifold_metrics['knn_distance'])[0,1]),
            'novelty_vs_reconstruction_error': float(np.corrcoef(novelties, offmanifold_metrics['reconstruction_error'])[0,1]),
        },
        'top_alien_indices': top_indices.tolist(),
    }
    
    with open(output_dir / "analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n  Saved: analysis_results.json")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    
    return analysis_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive MAP-Elites Analysis")
    
    parser.add_argument("--search-dir", type=str, required=True,
                        help="Path to MAP-Elites output directory")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to WikiArt reference embeddings (auto-detected if not specified)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        choices=["dino", "clip"],
                        help="Embedding model used (auto-detected from archive if not specified)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: search-dir/analysis)")
    
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    output_dir = Path(args.output_dir) if args.output_dir else search_dir / "analysis"
    
    # Auto-detect embedding model from archive_data.json
    embedding_model = args.embedding_model
    if embedding_model is None:
        archive_path = search_dir / "archive_data.json"
        if archive_path.exists():
            with open(archive_path) as f:
                archive_data = json.load(f)
            embedding_model = archive_data.get("config", {}).get("embedding_model", "dino")
        else:
            embedding_model = "dino"
        print(f"Auto-detected embedding model: {embedding_model}")
    
    # Auto-select reference based on embedding model
    if args.reference is None:
        ref_base = "/project/jevans/avi/wikiart_reference"
        if embedding_model == "clip":
            reference_path = Path(f"{ref_base}/wikiart_clip_81444.pkl")
        else:
            reference_path = Path(f"{ref_base}/wikiart_dino_81444.pkl")
        print(f"Using reference: {reference_path}")
    else:
        reference_path = Path(args.reference)
    
    run_full_analysis(search_dir, reference_path, output_dir)


if __name__ == "__main__":
    main()
