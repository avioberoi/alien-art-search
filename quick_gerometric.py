"""
Quick Geometric Off-Manifold Analysis
=====================================
Simplified version for quick analysis of existing results.

Usage:
  python quick_geometric.py \
      --wikiart-embeddings path/to/wikiart_embeddings.npy \
      --search-embeddings path/to/search_embeddings.npy \
      --output-dir outputs/geometric
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import json
import pickle
from tqdm import tqdm


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from various formats."""
    path = Path(path)
    
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            # Try common keys
            for key in ['embeddings', 'dino_embeddings', 'clip_embeddings']:
                if key in data:
                    emb = data[key]
                    break
            else:
                emb = list(data.values())[0]
        else:
            emb = data
        
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        return emb
    elif path.suffix == '.pt':
        data = torch.load(path)
        if isinstance(data, dict):
            data = list(data.values())[0]
        return data.numpy() if isinstance(data, torch.Tensor) else data
    else:
        raise ValueError(f"Unknown format: {path.suffix}")


class QuickGeometricAnalyzer:
    """Quick geometric off-manifold analyzer."""
    
    def __init__(self, reference_embeddings: np.ndarray, k: int = 10, pca_var: float = 0.95):
        """
        Initialize with reference embeddings (e.g., WikiArt).
        """
        self.reference = reference_embeddings
        self.n_ref, self.n_dims = reference_embeddings.shape
        self.k = k
        
        print(f"Fitting reference model on {self.n_ref} samples, {self.n_dims} dims...")
        
        # 1. Mean
        self.mean = reference_embeddings.mean(axis=0)
        centered = reference_embeddings - self.mean
        
        # 2. Full PCA
        print("  Fitting PCA...")
        self.pca = PCA()
        self.pca.fit(centered)
        self.eigenvalues = self.pca.explained_variance_
        
        # Find cutoff for requested variance
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = np.searchsorted(cumvar, pca_var) + 1
        print(f"  {self.n_components} components explain {pca_var*100:.0f}% variance")
        
        # 3. Ledoit-Wolf covariance
        print("  Fitting Ledoit-Wolf covariance...")
        lw = LedoitWolf().fit(reference_embeddings)
        self.precision = lw.get_precision()
        print(f"  Shrinkage: {lw.shrinkage_:.4f}")
        
        # 4. k-NN
        print(f"  Building k-NN index (k={k})...")
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        self.knn.fit(reference_embeddings)
        
        # Pre-compute reference statistics for percentiles
        print("  Computing reference statistics...")
        self.ref_mahal = self._mahalanobis(reference_embeddings)
        self.ref_low_pc_mag = self._low_pc_magnitude(reference_embeddings)
        self.ref_knn_dist = self.knn.kneighbors(reference_embeddings)[0].mean(axis=1)
        
        print("  Done!")
    
    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance."""
        delta = X - self.mean
        mahal_sq = np.einsum('ij,jk,ik->i', delta, self.precision, delta)
        return np.sqrt(np.maximum(mahal_sq, 0))
    
    def _low_pc_magnitude(self, X: np.ndarray, n_low: int = 10) -> np.ndarray:
        """Magnitude in lowest-variance PC directions."""
        centered = X - self.mean
        scores = self.pca.transform(centered)
        # Get lowest variance indices
        low_idx = np.argsort(self.eigenvalues)[:n_low]
        return np.linalg.norm(scores[:, low_idx], axis=1)
    
    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """PCA reconstruction error (SPE)."""
        centered = X - self.mean
        scores = self.pca.transform(centered)[:, :self.n_components]
        reconstructed = scores @ self.pca.components_[:self.n_components]
        return np.linalg.norm(centered - reconstructed, axis=1) ** 2
    
    def analyze(self, embeddings: np.ndarray) -> dict:
        """Compute all off-manifold metrics."""
        results = {}
        
        # Mahalanobis
        results['mahalanobis'] = self._mahalanobis(embeddings)
        results['mahalanobis_percentile'] = np.array([
            stats.percentileofscore(self.ref_mahal, m) for m in results['mahalanobis']
        ])
        
        # PCA
        results['reconstruction_error'] = self._reconstruction_error(embeddings)
        results['low_pc_magnitude'] = self._low_pc_magnitude(embeddings)
        results['low_pc_percentile'] = np.array([
            stats.percentileofscore(self.ref_low_pc_mag, m) for m in results['low_pc_magnitude']
        ])
        
        # k-NN
        results['knn_distance'] = self.knn.kneighbors(embeddings)[0].mean(axis=1)
        results['knn_percentile'] = np.array([
            stats.percentileofscore(self.ref_knn_dist, d) for d in results['knn_distance']
        ])
        
        # Combined score (geometric mean of percentiles, inverted for k-NN density)
        results['combined_score'] = (
            results['mahalanobis_percentile'] / 100 *
            results['low_pc_percentile'] / 100 *
            results['knn_percentile'] / 100
        ) ** (1/3)
        
        # Low-PC coordinates for visualization
        centered = embeddings - self.mean
        scores = self.pca.transform(centered)
        low_idx = np.argsort(self.eigenvalues)[:10]
        results['low_pc_coords'] = scores[:, low_idx]
        
        return results
    
    def get_reference_low_pc_coords(self, n_low: int = 2) -> np.ndarray:
        """Get WikiArt coordinates in low-variance PC space."""
        centered = self.reference - self.mean
        scores = self.pca.transform(centered)
        low_idx = np.argsort(self.eigenvalues)[:n_low]
        return scores[:, low_idx]


def visualize_comparison(
    results: dict,  # method_name -> analysis results
    analyzer: QuickGeometricAnalyzer,
    output_dir: Path,
    novelties: dict = None,  # method_name -> novelty scores (optional)
):
    """Create comprehensive comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results.keys())
    colors = {'random': '#3498db', 'cma': '#e74c3c', 'map_elites': '#2ecc71', 
              'map_elites_v2': '#9b59b6', 'wikiart': '#95a5a6'}
    
    # =========================================================================
    # Figure 1: Main Off-Manifold Metrics
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Geometric Off-Manifold Analysis: How Far from WikiArt?", fontsize=16, fontweight='bold')
    
    # 1a. Mahalanobis histogram
    ax = axes[0, 0]
    for method in methods:
        mahal = results[method]['mahalanobis']
        ax.hist(mahal, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(mahal):.2f})",
                color=colors.get(method, 'gray'))
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Count")
    ax.set_title("Mahalanobis Distance Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1b. Mahalanobis percentile
    ax = axes[0, 1]
    for method in methods:
        pct = results[method]['mahalanobis_percentile']
        ax.hist(pct, bins=20, alpha=0.5, label=f"{method} (μ={np.mean(pct):.1f}%)",
                color=colors.get(method, 'gray'))
    ax.axvline(95, color='red', linestyle='--', linewidth=2, label='95th percentile')
    ax.set_xlabel("Percentile vs WikiArt")
    ax.set_ylabel("Count")
    ax.set_title("How Extreme? (Mahalanobis Percentile)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1c. k-NN distance
    ax = axes[0, 2]
    for method in methods:
        knn = results[method]['knn_distance']
        ax.hist(knn, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(knn):.3f})",
                color=colors.get(method, 'gray'))
    ax.set_xlabel("Avg Distance to k Nearest WikiArt Images")
    ax.set_ylabel("Count")
    ax.set_title("k-NN Distance (Sparse Region Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1d. Low-variance PC magnitude
    ax = axes[1, 0]
    for method in methods:
        low = results[method]['low_pc_magnitude']
        ax.hist(low, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(low):.3f})",
                color=colors.get(method, 'gray'))
    ax.set_xlabel("Magnitude in Low-Variance PCs")
    ax.set_ylabel("Count")
    ax.set_title("Activity in Rare Directions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1e. Combined score
    ax = axes[1, 1]
    for method in methods:
        score = results[method]['combined_score']
        ax.hist(score, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(score):.3f})",
                color=colors.get(method, 'gray'))
    ax.set_xlabel("Combined Off-Manifold Score")
    ax.set_ylabel("Count")
    ax.set_title("Geometric Mean of All Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1f. Summary bar chart
    ax = axes[1, 2]
    metric_keys = ['mahalanobis', 'knn_distance', 'low_pc_magnitude']
    metric_labels = ['Mahalanobis', 'k-NN Dist', 'Low-PC Mag']
    x = np.arange(len(metric_keys))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        means = [np.mean(results[method][k]) for k in metric_keys]
        ax.bar(x + i * width, means, width, label=method, color=colors.get(method, 'gray'), alpha=0.7)
    
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Mean Value")
    ax.set_title("Summary Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "geometric_metrics.png", dpi=150)
    plt.close()
    print(f"  Saved: geometric_metrics.png")
    
    # =========================================================================
    # Figure 2: Low-Variance PC Space
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exploration of Low-Variance PC Space (Rarely-Occupied Directions)", 
                 fontsize=14, fontweight='bold')
    
    # WikiArt distribution
    wikiart_coords = analyzer.get_reference_low_pc_coords(2)
    
    # 2a. 2D scatter in low-PC space
    ax = axes[0]
    ax.scatter(wikiart_coords[:, 0], wikiart_coords[:, 1], 
               alpha=0.1, s=5, color='gray', label='WikiArt', rasterized=True)
    
    for method in methods:
        coords = results[method]['low_pc_coords']
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30,
                   label=method, color=colors.get(method, 'purple'))
    
    eigenvalues = analyzer.eigenvalues
    low_idx = np.argsort(eigenvalues)[:2]
    ax.set_xlabel(f"PC {low_idx[0]} (λ={eigenvalues[low_idx[0]]:.2e})")
    ax.set_ylabel(f"PC {low_idx[1]} (λ={eigenvalues[low_idx[1]]:.2e})")
    ax.set_title("Generated Images vs WikiArt in Rare PC Directions")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2b. Magnitude distribution
    ax = axes[1]
    wikiart_mags = np.linalg.norm(wikiart_coords, axis=1)
    ax.hist(wikiart_mags, bins=50, alpha=0.3, color='gray', label='WikiArt', density=True)
    
    for method in methods:
        mags = results[method]['low_pc_magnitude']
        ax.hist(mags, bins=30, alpha=0.5, color=colors.get(method, 'purple'), 
                label=method, density=True)
    
    ax.set_xlabel("Magnitude in Low-Variance PC Space")
    ax.set_ylabel("Density")
    ax.set_title("How Far from Origin in Rare Directions?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "low_pc_space.png", dpi=150)
    plt.close()
    print(f"  Saved: low_pc_space.png")
    
    # =========================================================================
    # Figure 3: Novelty vs Off-Manifold (if novelty provided)
    # =========================================================================
    if novelties:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Search Novelty vs Geometric Off-Manifold Metrics", fontsize=14, fontweight='bold')
        
        # 3a. Novelty vs Mahalanobis
        ax = axes[0]
        for method in methods:
            if method in novelties:
                nov = novelties[method]
                mahal = results[method]['mahalanobis']
                corr = np.corrcoef(nov, mahal)[0, 1]
                ax.scatter(nov, mahal, alpha=0.5, s=20, label=f"{method} (r={corr:.2f})",
                           color=colors.get(method, 'gray'))
        ax.set_xlabel("Search Novelty (vs History)")
        ax.set_ylabel("Mahalanobis Distance (vs WikiArt)")
        ax.set_title("Does Search Novelty → Off-Manifold?")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3b. Novelty vs k-NN distance
        ax = axes[1]
        for method in methods:
            if method in novelties:
                nov = novelties[method]
                knn = results[method]['knn_distance']
                corr = np.corrcoef(nov, knn)[0, 1]
                ax.scatter(nov, knn, alpha=0.5, s=20, label=f"{method} (r={corr:.2f})",
                           color=colors.get(method, 'gray'))
        ax.set_xlabel("Search Novelty (vs History)")
        ax.set_ylabel("k-NN Distance (vs WikiArt)")
        ax.set_title("Search Novelty vs Sparse Region Discovery")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3c. Novelty vs Combined score
        ax = axes[2]
        for method in methods:
            if method in novelties:
                nov = novelties[method]
                score = results[method]['combined_score']
                corr = np.corrcoef(nov, score)[0, 1]
                ax.scatter(nov, score, alpha=0.5, s=20, label=f"{method} (r={corr:.2f})",
                           color=colors.get(method, 'gray'))
        ax.set_xlabel("Search Novelty (vs History)")
        ax.set_ylabel("Combined Off-Manifold Score")
        ax.set_title("Overall Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "novelty_vs_geometric.png", dpi=150)
        plt.close()
        print(f"  Saved: novelty_vs_geometric.png")
    
    # =========================================================================
    # Print Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("GEOMETRIC OFF-MANIFOLD ANALYSIS SUMMARY")
    print("=" * 80)
    
    header = f"{'Metric':<35} " + " ".join([f"{m:<15}" for m in methods])
    print(header)
    print("-" * len(header))
    
    metrics_to_print = [
        ('mahalanobis', 'Mahalanobis Distance'),
        ('mahalanobis_percentile', 'Mahalanobis Percentile (%)'),
        ('knn_distance', 'k-NN Distance'),
        ('knn_percentile', 'k-NN Percentile (%)'),
        ('low_pc_magnitude', 'Low-PC Magnitude'),
        ('low_pc_percentile', 'Low-PC Percentile (%)'),
        ('reconstruction_error', 'PCA Reconstruction Error'),
        ('combined_score', 'Combined Off-Manifold Score'),
    ]
    
    summary_data = {}
    for key, name in metrics_to_print:
        values = [np.mean(results[m][key]) for m in methods]
        print(f"{name:<35} " + " ".join([f"{v:<15.4f}" for v in values]))
        summary_data[key] = {m: float(np.mean(results[m][key])) for m in methods}
    
    # Determine winner for each metric
    print("\n" + "-" * 80)
    print("HIGHER = MORE OFF-MANIFOLD (better for alien art discovery)")
    print("-" * 80)
    
    for key, name in metrics_to_print:
        values = {m: np.mean(results[m][key]) for m in methods}
        winner = max(values, key=values.get)
        print(f"  {name}: {winner} wins ({values[winner]:.4f})")
    
    # Save summary
    with open(output_dir / "geometric_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return summary_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick Geometric Off-Manifold Analysis")
    
    parser.add_argument("--wikiart", type=str, required=True,
                        help="Path to WikiArt embeddings")
    parser.add_argument("--output-dir", type=str, default="outputs/geometric",
                        help="Output directory")
    
    # Can specify multiple search results
    parser.add_argument("--random", type=str, help="Random search embeddings")
    parser.add_argument("--cma", type=str, help="CMA-ES embeddings")
    parser.add_argument("--mapelites", type=str, help="MAP-Elites embeddings")
    parser.add_argument("--mapelites-v2", type=str, help="MAP-Elites v2 embeddings")
    
    # Novelty scores (optional, for correlation analysis)
    parser.add_argument("--random-novelty", type=str, help="Random novelty scores")
    parser.add_argument("--cma-novelty", type=str, help="CMA novelty scores")
    parser.add_argument("--mapelites-novelty", type=str, help="MAP-Elites novelty scores")
    
    parser.add_argument("--k", type=int, default=10, help="k for k-NN")
    parser.add_argument("--pca-var", type=float, default=0.95, help="PCA variance threshold")
    
    args = parser.parse_args()
    
    # Load WikiArt
    print(f"Loading WikiArt embeddings from {args.wikiart}...")
    wikiart = load_embeddings(args.wikiart)
    print(f"  Shape: {wikiart.shape}")
    
    # Initialize analyzer
    analyzer = QuickGeometricAnalyzer(wikiart, k=args.k, pca_var=args.pca_var)
    
    # Load and analyze each method
    results = {}
    novelties = {}
    
    method_args = [
        ('random', args.random, args.random_novelty),
        ('cma', args.cma, args.cma_novelty),
        ('map_elites', args.mapelites, args.mapelites_novelty),
        ('map_elites_v2', args.mapelites_v2, None),
    ]
    
    for method_name, emb_path, nov_path in method_args:
        if emb_path:
            print(f"\nAnalyzing {method_name}...")
            embeddings = load_embeddings(emb_path)
            print(f"  Shape: {embeddings.shape}")
            results[method_name] = analyzer.analyze(embeddings)
            
            if nov_path:
                novelties[method_name] = load_embeddings(nov_path)
    
    if not results:
        print("No search results provided!")
        return
    
    # Visualize
    print("\nGenerating visualizations...")
    output_dir = Path(args.output_dir)
    visualize_comparison(results, analyzer, output_dir, novelties if novelties else None)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()