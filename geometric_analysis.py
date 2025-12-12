"""
Geometric Off-Manifold Analysis for Alien Art Discovery
=========================================================
Measures how "off-manifold" generated images are relative to WikiArt reference
using multiple geometric metrics:

1. Mahalanobis Distance (with Ledoit-Wolf regularization)
2. PCA Reconstruction Error (SPE) + Hotelling's T²
3. Low-Variance PC Coordinates (directions WikiArt rarely uses)
4. k-NN Density Estimation (non-parametric)

Usage:
  # Analyze existing search results
  python geometric_analysis.py --search-dir outputs/map_elites --reference wikiart_embeddings.pkl
  
  # Compare multiple methods
  python geometric_analysis.py --compare \
      --random-dir outputs/search \
      --cma-dir outputs/cma_search \
      --mapelites-dir outputs/map_elites \
      --reference wikiart_embeddings.pkl
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pickle
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeometricConfig:
    """Configuration for geometric analysis."""
    # PCA settings
    pca_variance_threshold: float = 0.95  # Keep components explaining this much variance
    n_low_variance_pcs: int = 10  # Number of low-variance PCs to analyze
    
    # k-NN settings
    k_neighbors: int = 10  # For density estimation
    
    # Mahalanobis settings
    regularization: str = "ledoit_wolf"  # or "ridge", "pca"
    ridge_alpha: float = 0.01  # For ridge regularization
    
    # Output
    output_dir: Path = Path("outputs/geometric_analysis")
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REFERENCE MODEL (fits to WikiArt embeddings)
# =============================================================================

class GeometricReferenceModel:
    """
    Statistical model of the reference distribution (WikiArt).
    Fits PCA, computes covariance, and enables multiple off-manifold metrics.
    """
    
    def __init__(self, config: GeometricConfig):
        self.config = config
        self.fitted = False
        
        # Will be set after fitting
        self.mean = None
        self.pca = None
        self.pca_full = None  # Full PCA for low-variance analysis
        self.precision_matrix = None  # For Mahalanobis
        self.knn_model = None
        self.reference_embeddings = None
        
        # Statistics
        self.n_samples = 0
        self.n_dims = 0
        self.n_components = 0
        self.explained_variance_ratio = None
        self.eigenvalues = None
        
    def fit(self, embeddings: np.ndarray):
        """
        Fit the reference model to WikiArt embeddings.
        
        Args:
            embeddings: [N, D] array of normalized embeddings
        """
        print("Fitting geometric reference model...")
        self.reference_embeddings = embeddings
        self.n_samples, self.n_dims = embeddings.shape
        
        # 1. Compute mean
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        
        # 2. Fit full PCA (for eigenvalue analysis)
        print(f"  Fitting full PCA on {self.n_samples} samples, {self.n_dims} dims...")
        self.pca_full = PCA(n_components=min(self.n_samples, self.n_dims))
        self.pca_full.fit(centered)
        self.eigenvalues = self.pca_full.explained_variance_
        self.explained_variance_ratio = self.pca_full.explained_variance_ratio_
        
        # 3. Fit reduced PCA (for reconstruction error)
        cumvar = np.cumsum(self.explained_variance_ratio)
        self.n_components = np.searchsorted(cumvar, self.config.pca_variance_threshold) + 1
        self.n_components = min(self.n_components, self.n_dims - 1)
        print(f"  Using {self.n_components} components ({self.config.pca_variance_threshold*100:.0f}% variance)")
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(centered)
        
        # 4. Fit covariance model for Mahalanobis
        print(f"  Fitting Ledoit-Wolf covariance estimator...")
        lw = LedoitWolf()
        lw.fit(embeddings)
        self.precision_matrix = lw.get_precision()
        self.covariance_matrix = lw.covariance_
        print(f"  Shrinkage coefficient: {lw.shrinkage_:.4f}")
        
        # 5. Fit k-NN model for density estimation
        print(f"  Building k-NN index (k={self.config.k_neighbors})...")
        self.knn_model = NearestNeighbors(
            n_neighbors=self.config.k_neighbors,
            metric='cosine',  # For normalized embeddings
            algorithm='auto',
        )
        self.knn_model.fit(embeddings)
        
        self.fitted = True
        print("  Reference model fitted!")
        
        return self
    
    def compute_mahalanobis(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance from reference distribution.
        
        Returns: [N] array of Mahalanobis distances
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        delta = embeddings - self.mean
        # D_M^2 = delta @ precision @ delta.T (diagonal)
        # Using einsum for efficiency
        mahal_sq = np.einsum('ij,jk,ik->i', delta, self.precision_matrix, delta)
        return np.sqrt(np.maximum(mahal_sq, 0))  # Ensure non-negative
    
    def compute_pca_scores(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute PCA-based anomaly scores.
        
        Returns dict with:
            - reconstruction_error (SPE): Distance to PCA subspace
            - hotelling_t2: Deviation within PCA subspace
            - low_pc_coords: Coordinates on low-variance PCs [N, n_low_pcs]
            - low_pc_magnitude: L2 norm of low-variance PC coordinates
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        centered = embeddings - self.mean
        
        # Project to reduced PCA space and back
        scores = self.pca.transform(centered)
        reconstructed = self.pca.inverse_transform(scores)
        
        # 1. Reconstruction error (SPE)
        reconstruction_error = np.linalg.norm(centered - reconstructed, axis=1) ** 2
        
        # 2. Hotelling's T² (within subspace)
        # T² = sum(score_i² / eigenvalue_i)
        eigenvalues = self.pca.explained_variance_
        hotelling_t2 = np.sum(scores ** 2 / eigenvalues, axis=1)
        
        # 3. Low-variance PC coordinates (using full PCA)
        full_scores = self.pca_full.transform(centered)
        
        # Get indices of lowest-variance PCs
        n_low = self.config.n_low_variance_pcs
        low_pc_indices = np.argsort(self.eigenvalues)[:n_low]
        low_pc_coords = full_scores[:, low_pc_indices]
        
        # 4. Magnitude in low-variance directions
        low_pc_magnitude = np.linalg.norm(low_pc_coords, axis=1)
        
        # Also compute normalized magnitude (relative to WikiArt distribution)
        # This tells us how extreme we are compared to typical WikiArt
        wikiart_low_coords = self.pca_full.transform(self.reference_embeddings - self.mean)[:, low_pc_indices]
        wikiart_low_mags = np.linalg.norm(wikiart_low_coords, axis=1)
        low_pc_percentile = np.array([
            stats.percentileofscore(wikiart_low_mags, mag) for mag in low_pc_magnitude
        ])
        
        return {
            'reconstruction_error': reconstruction_error,
            'hotelling_t2': hotelling_t2,
            'low_pc_coords': low_pc_coords,
            'low_pc_magnitude': low_pc_magnitude,
            'low_pc_percentile': low_pc_percentile,
            'low_pc_indices': low_pc_indices,
        }
    
    def compute_knn_density(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute k-NN based density scores.
        
        Returns dict with:
            - knn_distance: Average distance to k nearest neighbors
            - knn_density: Inverse density estimate (higher = more anomalous)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        distances, indices = self.knn_model.kneighbors(embeddings)
        
        # Average distance to k neighbors
        knn_distance = distances.mean(axis=1)
        
        # Density estimate (inverse of average distance)
        # Add small epsilon to avoid division by zero
        knn_density_score = 1.0 / (knn_distance + 1e-8)
        
        return {
            'knn_distance': knn_distance,
            'knn_density': knn_density_score,
            'knn_distances_all': distances,
        }
    
    def compute_all_metrics(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all off-manifold metrics at once."""
        results = {
            'mahalanobis': self.compute_mahalanobis(embeddings),
        }
        results.update(self.compute_pca_scores(embeddings))
        results.update(self.compute_knn_density(embeddings))
        
        # Compute combined "alien score"
        # Normalize each metric to [0, 1] range based on WikiArt distribution
        # Then combine multiplicatively
        
        # For Mahalanobis: use chi-squared percentile
        # Pre-compute reference Mahalanobis distances ONCE (cached)
        if not hasattr(self, '_ref_mahal_cached'):
            print("  Computing reference Mahalanobis distances (one-time)...")
            self._ref_mahal_cached = self.compute_mahalanobis(self.reference_embeddings)
        
        mahal_percentile = np.array([
            stats.percentileofscore(self._ref_mahal_cached, m) 
            for m in results['mahalanobis']
        ])
        results['mahalanobis_percentile'] = mahal_percentile
        
        # Combined score: geometric mean of percentiles
        combined = (
            mahal_percentile / 100 * 
            results['low_pc_percentile'] / 100 * 
            (1 - np.clip(results['knn_density'] / results['knn_density'].max(), 0, 1))
        ) ** (1/3)
        results['combined_alien_score'] = combined
        
        return results
    
    def save(self, path: Path):
        """Save fitted model to disk."""
        data = {
            'mean': self.mean,
            'precision_matrix': self.precision_matrix,
            'covariance_matrix': self.covariance_matrix,
            'pca_components': self.pca.components_,
            'pca_explained_variance': self.pca.explained_variance_,
            'pca_mean': self.pca.mean_,
            'pca_full_components': self.pca_full.components_,
            'pca_full_explained_variance': self.pca_full.explained_variance_,
            'eigenvalues': self.eigenvalues,
            'explained_variance_ratio': self.explained_variance_ratio,
            'n_samples': self.n_samples,
            'n_dims': self.n_dims,
            'n_components': self.n_components,
            'config': self.config,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, reference_embeddings: np.ndarray = None) -> "GeometricReferenceModel":
        """Load fitted model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['config'])
        model.mean = data['mean']
        model.precision_matrix = data['precision_matrix']
        model.covariance_matrix = data['covariance_matrix']
        model.eigenvalues = data['eigenvalues']
        model.explained_variance_ratio = data['explained_variance_ratio']
        model.n_samples = data['n_samples']
        model.n_dims = data['n_dims']
        model.n_components = data['n_components']
        
        # Reconstruct PCA objects
        model.pca = PCA(n_components=model.n_components)
        model.pca.components_ = data['pca_components']
        model.pca.explained_variance_ = data['pca_explained_variance']
        model.pca.mean_ = data['pca_mean']
        
        model.pca_full = PCA(n_components=len(data['eigenvalues']))
        model.pca_full.components_ = data['pca_full_components']
        model.pca_full.explained_variance_ = data['pca_full_explained_variance']
        model.pca_full.mean_ = data['pca_mean']
        
        # Rebuild k-NN if reference provided
        if reference_embeddings is not None:
            model.reference_embeddings = reference_embeddings
            model.knn_model = NearestNeighbors(
                n_neighbors=model.config.k_neighbors,
                metric='cosine',
            )
            model.knn_model.fit(reference_embeddings)
        
        model.fitted = True
        return model


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_embeddings_from_file(emb_path: Path) -> np.ndarray:
    """
    Load pre-computed embeddings from a file (.pkl or .npy).
    """
    emb_path = Path(emb_path)
    
    if emb_path.suffix == '.npy':
        return np.load(emb_path)
    elif emb_path.suffix == '.pkl':
        with open(emb_path, 'rb') as f:
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
    elif emb_path.suffix == '.pt':
        data = torch.load(emb_path)
        if isinstance(data, dict):
            data = list(data.values())[0]
        return data.numpy() if isinstance(data, torch.Tensor) else data
    else:
        raise ValueError(f"Unknown embedding format: {emb_path.suffix}")


def load_novelties_from_search(search_dir: Path) -> np.ndarray:
    """
    Load novelty scores from a search output directory's JSON log.
    """
    search_dir = Path(search_dir)
    
    # Try different possible file names
    possible_files = [
        search_dir / "archive_data.json",
        search_dir / "search_log.json", 
        search_dir / "cma_log.json",
    ]
    
    log_file = None
    for f in possible_files:
        if f.exists():
            log_file = f
            break
    
    if log_file is None:
        print(f"  Warning: No log file found in {search_dir}")
        return None
    
    with open(log_file) as f:
        data = json.load(f)
    
    results = data.get("results", data.get("elites", []))
    novelties = np.array([r.get("novelty", 0) for r in results])
    
    return novelties


def load_embeddings_from_search(search_dir: Path, embedding_type: str = "dino") -> Tuple[np.ndarray, List[dict]]:
    """
    Load embeddings from a search output directory.
    
    Returns:
        embeddings: [N, D] array
        metadata: List of dicts with image info
    """
    # Try different possible file names
    possible_files = [
        search_dir / "archive_data.json",
        search_dir / "search_log.json", 
        search_dir / "cma_log.json",
    ]
    
    log_file = None
    for f in possible_files:
        if f.exists():
            log_file = f
            break
    
    if log_file is None:
        raise FileNotFoundError(f"No log file found in {search_dir}")
    
    with open(log_file) as f:
        data = json.load(f)
    
    results = data.get("results", data.get("elites", []))
    
    # Load embeddings from images
    embeddings = []
    metadata = []
    
    # Check if embeddings are stored in the log
    if results and "dino_embedding" in results[0]:
        # Embeddings stored in log
        for r in results:
            emb = np.array(r.get(f"{embedding_type}_embedding", r.get("dino_embedding")))
            embeddings.append(emb)
            metadata.append(r)
    else:
        # Need to re-embed images
        print(f"  Embeddings not in log, need to re-compute from images...")
        # This would require loading the embedding model
        raise NotImplementedError("Re-embedding not implemented. Use --mapelites-emb etc. for pre-computed embeddings.")
    
    return np.array(embeddings), metadata


def analyze_search_results(
    search_dir: Path,
    reference_model: GeometricReferenceModel,
    config: GeometricConfig,
) -> Dict:
    """
    Analyze search results using the geometric reference model.
    """
    print(f"\nAnalyzing {search_dir}...")
    
    # Load embeddings
    embeddings, metadata = load_embeddings_from_search(search_dir)
    print(f"  Loaded {len(embeddings)} embeddings")
    
    # Compute all metrics
    metrics = reference_model.compute_all_metrics(embeddings)
    
    # Compile statistics
    stats_dict = {}
    for key, values in metrics.items():
        if isinstance(values, np.ndarray) and values.ndim == 1:
            stats_dict[f"{key}_mean"] = float(np.mean(values))
            stats_dict[f"{key}_std"] = float(np.std(values))
            stats_dict[f"{key}_median"] = float(np.median(values))
            stats_dict[f"{key}_min"] = float(np.min(values))
            stats_dict[f"{key}_max"] = float(np.max(values))
    
    # Add search-specific novelty if available
    if metadata and "novelty" in metadata[0]:
        novelties = [m.get("novelty", 0) for m in metadata]
        stats_dict["search_novelty_mean"] = float(np.mean(novelties))
        stats_dict["search_novelty_std"] = float(np.std(novelties))
        metrics["search_novelty"] = np.array(novelties)
    
    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "metrics": metrics,
        "stats": stats_dict,
    }


def compare_methods(
    results_dict: Dict[str, Dict],
    reference_model: GeometricReferenceModel,
    output_dir: Path,
):
    """
    Create comparison visualizations across methods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results_dict.keys())
    colors = {'random': 'blue', 'cma': 'orange', 'map_elites': 'green', 'wikiart': 'gray'}
    
    # =========================================================================
    # 1. Mahalanobis Distance Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mahalanobis Distance: How Far from WikiArt Distribution?", fontsize=14)
    
    # 1a. Histograms
    ax = axes[0]
    for method, data in results_dict.items():
        mahal = data['metrics']['mahalanobis']
        ax.hist(mahal, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(mahal):.2f})",
                color=colors.get(method, 'purple'))
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mahalanobis Distances")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1b. Box plot
    ax = axes[1]
    box_data = [data['metrics']['mahalanobis'] for data in results_dict.values()]
    bp = ax.boxplot(box_data, labels=methods, patch_artist=True)
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, 'purple'))
        patch.set_alpha(0.5)
    ax.set_ylabel("Mahalanobis Distance")
    ax.set_title("Comparison Across Methods")
    ax.grid(True, alpha=0.3)
    
    # 1c. Percentile plot
    ax = axes[2]
    for method, data in results_dict.items():
        percentiles = data['metrics']['mahalanobis_percentile']
        ax.hist(percentiles, bins=20, alpha=0.5, label=method, color=colors.get(method, 'purple'))
    ax.axvline(95, color='red', linestyle='--', label='95th percentile')
    ax.set_xlabel("Percentile (vs WikiArt)")
    ax.set_ylabel("Count")
    ax.set_title("How Extreme? (Percentile of Mahalanobis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mahalanobis_comparison.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 2. PCA-based Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PCA-Based Off-Manifold Analysis", fontsize=14)
    
    # 2a. Reconstruction error
    ax = axes[0, 0]
    for method, data in results_dict.items():
        spe = data['metrics']['reconstruction_error']
        ax.hist(spe, bins=30, alpha=0.5, label=f"{method}", color=colors.get(method, 'purple'))
    ax.set_xlabel("Reconstruction Error (SPE)")
    ax.set_ylabel("Count")
    ax.set_title("Distance from PCA Subspace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2b. Hotelling's T²
    ax = axes[0, 1]
    for method, data in results_dict.items():
        t2 = data['metrics']['hotelling_t2']
        ax.hist(np.log10(t2 + 1), bins=30, alpha=0.5, label=method, color=colors.get(method, 'purple'))
    ax.set_xlabel("log₁₀(Hotelling's T² + 1)")
    ax.set_ylabel("Count")
    ax.set_title("Deviation Within PCA Subspace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2c. Low-variance PC magnitude
    ax = axes[0, 2]
    for method, data in results_dict.items():
        low_mag = data['metrics']['low_pc_magnitude']
        ax.hist(low_mag, bins=30, alpha=0.5, label=method, color=colors.get(method, 'purple'))
    ax.set_xlabel("Magnitude in Low-Variance PCs")
    ax.set_ylabel("Count")
    ax.set_title("Activity in Rare Directions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2d. Low-variance PC percentile
    ax = axes[1, 0]
    for method, data in results_dict.items():
        pct = data['metrics']['low_pc_percentile']
        ax.hist(pct, bins=20, alpha=0.5, label=method, color=colors.get(method, 'purple'))
    ax.axvline(95, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Percentile (Low-Variance PC Magnitude)")
    ax.set_ylabel("Count")
    ax.set_title("How Extreme in Rare Directions?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2e. Scatter: SPE vs T²
    ax = axes[1, 1]
    for method, data in results_dict.items():
        spe = data['metrics']['reconstruction_error']
        t2 = data['metrics']['hotelling_t2']
        ax.scatter(np.log10(t2 + 1), np.log10(spe + 1), alpha=0.5, s=20, 
                   label=method, color=colors.get(method, 'purple'))
    ax.set_xlabel("log₁₀(T² + 1)")
    ax.set_ylabel("log₁₀(SPE + 1)")
    ax.set_title("Within-Subspace vs Orthogonal Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2f. Eigenvalue spectrum
    ax = axes[1, 2]
    eigenvalues = reference_model.eigenvalues
    ax.semilogy(eigenvalues, 'b-', linewidth=1)
    ax.axhline(eigenvalues[reference_model.n_components], color='red', linestyle='--',
               label=f'Cutoff ({reference_model.n_components} PCs)')
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("WikiArt Eigenvalue Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_analysis.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 3. k-NN Density Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("k-NN Density Analysis: Distance from WikiArt Samples", fontsize=14)
    
    # 3a. k-NN distance histogram
    ax = axes[0]
    for method, data in results_dict.items():
        knn_dist = data['metrics']['knn_distance']
        ax.hist(knn_dist, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(knn_dist):.3f})",
                color=colors.get(method, 'purple'))
    ax.set_xlabel("Average Distance to k Nearest WikiArt Images")
    ax.set_ylabel("Count")
    ax.set_title(f"k-NN Distance (k={reference_model.config.k_neighbors})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3b. Box plot
    ax = axes[1]
    box_data = [data['metrics']['knn_distance'] for data in results_dict.values()]
    bp = ax.boxplot(box_data, labels=methods, patch_artist=True)
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, 'purple'))
        patch.set_alpha(0.5)
    ax.set_ylabel("k-NN Distance")
    ax.set_title("Distance Comparison")
    ax.grid(True, alpha=0.3)
    
    # 3c. Scatter: k-NN distance vs search novelty (if available)
    ax = axes[2]
    for method, data in results_dict.items():
        knn_dist = data['metrics']['knn_distance']
        if 'search_novelty' in data['metrics']:
            novelty = data['metrics']['search_novelty']
            ax.scatter(novelty, knn_dist, alpha=0.5, s=20, label=method,
                       color=colors.get(method, 'purple'))
    ax.set_xlabel("Search Novelty (vs History)")
    ax.set_ylabel("k-NN Distance (vs WikiArt)")
    ax.set_title("Search Novelty vs WikiArt Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "knn_analysis.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 4. Low-Variance PC Space Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exploration of Low-Variance PC Space (Rare Directions)", fontsize=14)
    
    # 4a. WikiArt distribution in low-variance PCs
    ax = axes[0]
    wikiart_low = reference_model.pca_full.transform(
        reference_model.reference_embeddings - reference_model.mean
    )
    low_indices = np.argsort(reference_model.eigenvalues)[:2]  # Two lowest
    
    ax.scatter(wikiart_low[:, low_indices[0]], wikiart_low[:, low_indices[1]], 
               alpha=0.2, s=10, color='gray', label='WikiArt')
    
    for method, data in results_dict.items():
        low_coords = data['metrics']['low_pc_coords']
        ax.scatter(low_coords[:, 0], low_coords[:, 1], alpha=0.6, s=30,
                   label=method, color=colors.get(method, 'purple'))
    
    ax.set_xlabel(f"PC {low_indices[0]} (λ={reference_model.eigenvalues[low_indices[0]]:.4f})")
    ax.set_ylabel(f"PC {low_indices[1]} (λ={reference_model.eigenvalues[low_indices[1]]:.4f})")
    ax.set_title("Two Lowest-Variance Principal Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4b. Distribution of distances from origin in low-PC space
    ax = axes[1]
    
    # WikiArt baseline
    wikiart_mags = np.linalg.norm(wikiart_low[:, low_indices[:reference_model.config.n_low_variance_pcs]], axis=1)
    ax.hist(wikiart_mags, bins=30, alpha=0.3, color='gray', label='WikiArt', density=True)
    
    for method, data in results_dict.items():
        mags = data['metrics']['low_pc_magnitude']
        ax.hist(mags, bins=30, alpha=0.5, color=colors.get(method, 'purple'), 
                label=method, density=True)
    
    ax.set_xlabel("Magnitude in Low-Variance PC Space")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution in {reference_model.config.n_low_variance_pcs} Lowest-Variance PCs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "low_variance_pc_space.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 5. Combined Alien Score
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Combined Off-Manifold Score", fontsize=14)
    
    # 5a. Histogram of combined scores
    ax = axes[0]
    for method, data in results_dict.items():
        score = data['metrics']['combined_alien_score']
        ax.hist(score, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(score):.3f})",
                color=colors.get(method, 'purple'))
    ax.set_xlabel("Combined Alien Score")
    ax.set_ylabel("Count")
    ax.set_title("Geometric Mean of Off-Manifold Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5b. Summary bar chart
    ax = axes[1]
    metric_names = ['mahalanobis', 'reconstruction_error', 'low_pc_magnitude', 'knn_distance']
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    
    for i, (method, data) in enumerate(results_dict.items()):
        means = [np.mean(data['metrics'][m]) for m in metric_names]
        # Normalize by WikiArt mean for comparison
        ax.bar(x + i * width, means, width, label=method, color=colors.get(method, 'purple'), alpha=0.7)
    
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(['Mahalanobis', 'Recon. Error', 'Low-PC Mag', 'k-NN Dist'])
    ax.set_ylabel("Mean Value")
    ax.set_title("Off-Manifold Metrics Summary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_score.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 6. Summary Statistics Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("GEOMETRIC OFF-MANIFOLD ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    metrics_to_compare = [
        ('mahalanobis_mean', 'Mahalanobis Distance'),
        ('mahalanobis_percentile_mean', 'Mahalanobis Percentile'),
        ('reconstruction_error_mean', 'PCA Reconstruction Error'),
        ('low_pc_magnitude_mean', 'Low-Variance PC Magnitude'),
        ('low_pc_percentile_mean', 'Low-PC Percentile'),
        ('knn_distance_mean', 'k-NN Distance'),
        ('combined_alien_score_mean', 'Combined Alien Score'),
    ]
    
    print(f"\n{'Metric':<30} " + " ".join([f"{m:<15}" for m in methods]))
    print("-" * (30 + 16 * len(methods)))
    
    for metric_key, metric_name in metrics_to_compare:
        values = [results_dict[m]['stats'].get(metric_key, 0) for m in methods]
        print(f"{metric_name:<30} " + " ".join([f"{v:<15.4f}" for v in values]))
    
    # Save summary JSON
    summary = {
        "methods": methods,
        "metrics": {method: data['stats'] for method, data in results_dict.items()},
    }
    
    with open(output_dir / "geometric_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return summary


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geometric Off-Manifold Analysis")
    
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to WikiArt embeddings (.pkl or .npy)")
    parser.add_argument("--output-dir", type=str, default="outputs/geometric_analysis",
                        help="Output directory")
    
    # Single analysis mode
    parser.add_argument("--search-dir", type=str,
                        help="Path to search results directory")
    
    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple methods")
    parser.add_argument("--random-dir", type=str,
                        help="Path to random search results (for novelty scores)")
    parser.add_argument("--cma-dir", type=str,
                        help="Path to CMA-ES search results (for novelty scores)")
    parser.add_argument("--mapelites-dir", type=str,
                        help="Path to MAP-Elites results (for novelty scores)")
    
    # Pre-computed embeddings (use instead of extracting from search dirs)
    parser.add_argument("--random-emb", type=str,
                        help="Path to pre-computed random search embeddings (.pkl or .npy)")
    parser.add_argument("--cma-emb", type=str,
                        help="Path to pre-computed CMA-ES embeddings (.pkl or .npy)")
    parser.add_argument("--mapelites-emb", type=str,
                        help="Path to pre-computed MAP-Elites embeddings (.pkl or .npy)")
    
    # Model settings
    parser.add_argument("--pca-variance", type=float, default=0.95,
                        help="Variance threshold for PCA")
    parser.add_argument("--k-neighbors", type=int, default=10,
                        help="k for k-NN density estimation")
    
    args = parser.parse_args()
    
    config = GeometricConfig(
        pca_variance_threshold=args.pca_variance,
        k_neighbors=args.k_neighbors,
        output_dir=Path(args.output_dir),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference embeddings
    print(f"Loading reference embeddings from {args.reference}...", flush=True)
    ref_path = Path(args.reference)
    if ref_path.suffix == '.pkl':
        with open(ref_path, 'rb') as f:
            ref_data = pickle.load(f)
        if isinstance(ref_data, dict):
            reference_embeddings = ref_data.get('embeddings', ref_data.get('dino_embeddings'))
        else:
            reference_embeddings = ref_data
    else:
        reference_embeddings = np.load(ref_path)
    
    if isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = reference_embeddings.numpy()
    
    print(f"  Reference shape: {reference_embeddings.shape}", flush=True)
    
    # Fit reference model
    print("Fitting reference model...", flush=True)
    model = GeometricReferenceModel(config)
    model.fit(reference_embeddings)
    print("  Model fitted.", flush=True)
    
    # Save model
    model.save(config.output_dir / "reference_model.pkl")
    
    if args.compare:
        # Comparison mode
        results_dict = {}
        
        # Load pre-computed embeddings if provided, otherwise load from search dirs
        method_configs = [
            ('random', args.random_emb, args.random_dir),
            ('cma', args.cma_emb, args.cma_dir),
            ('map_elites', args.mapelites_emb, args.mapelites_dir),
        ]
        
        for method_name, emb_path, search_dir in method_configs:
            if emb_path:
                # Use pre-computed embeddings
                print(f"\nLoading pre-computed {method_name} embeddings from {emb_path}...", flush=True)
                embeddings = load_embeddings_from_file(emb_path)
                print(f"  Shape: {embeddings.shape}", flush=True)
                
                # Load novelty scores from search dir if available
                novelties = None
                if search_dir:
                    novelties = load_novelties_from_search(Path(search_dir))
                    if novelties is not None:
                        print(f"  Loaded {len(novelties)} novelty scores", flush=True)
                        # Truncate to match embedding count if needed
                        if len(novelties) > len(embeddings):
                            novelties = novelties[:len(embeddings)]
                
                # Compute metrics
                print(f"  Computing metrics for {method_name}...", flush=True)
                metrics = model.compute_all_metrics(embeddings)
                print(f"  Done computing metrics.", flush=True)
                
                # Add novelty if available
                if novelties is not None and len(novelties) == len(embeddings):
                    metrics['search_novelty'] = novelties
                
                # Compile stats
                stats_dict = {}
                for key, values in metrics.items():
                    stats_dict[f"{key}_mean"] = float(np.mean(values))
                    stats_dict[f"{key}_std"] = float(np.std(values))
                    stats_dict[f"{key}_min"] = float(np.min(values))
                    stats_dict[f"{key}_max"] = float(np.max(values))
                
                results_dict[method_name] = {
                    'embeddings': embeddings,
                    'metrics': metrics,
                    'stats': stats_dict,
                    'metadata': [],  # No metadata when using pre-computed
                }
            elif search_dir:
                # Load from search directory (requires embeddings in log)
                try:
                    results_dict[method_name] = analyze_search_results(Path(search_dir), model, config)
                except NotImplementedError as e:
                    print(f"  Skipping {method_name}: {e}")
        
        if len(results_dict) > 0:
            compare_methods(results_dict, model, config.output_dir)
        else:
            print("No embeddings provided for comparison!")
            print("Use --mapelites-emb, --cma-emb, --random-emb for pre-computed embeddings")
    
    elif args.search_dir:
        # Single analysis mode
        results = analyze_search_results(Path(args.search_dir), model, config)
        
        # Create visualizations for single method
        compare_methods({'search': results}, model, config.output_dir)
    
    else:
        print("Please provide --search-dir or --compare with method directories/embeddings")


if __name__ == "__main__":
    main()