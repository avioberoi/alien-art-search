#!/bin/bash
#SBATCH --job-name=geom_full
#SBATCH --account=pi-jevans
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/geometric_full_%j.out
#SBATCH --error=logs/geometric_full_%j.err

# ============================================================================
# Full Geometric Off-Manifold Analysis (CPU-only, v2)
# ============================================================================
# Uses pre-extracted embeddings from geometric_data/
# Runs comprehensive analysis with ALL metrics from geometric_analysis.py:
#   - Mahalanobis Distance (Ledoit-Wolf regularization)
#   - PCA Reconstruction Error (SPE)
#   - Hotelling's T² (within-subspace deviation)
#   - Low-Variance PC Magnitude + Percentile
#   - k-NN Density Estimation
#   - Combined Alien Score
# 
# Generates 5 visualization files:
#   1. mahalanobis_comparison.png - Distribution of Mahalanobis distances
#   2. pca_analysis.png - SPE, T², eigenvalues, low-PC metrics
#   3. knn_analysis.png - k-NN distance from WikiArt samples  
#   4. low_variance_pc_space.png - Exploration of rare directions
#   5. combined_score.png - Combined alien score summary
#
# Also includes novelty correlation analysis
# ============================================================================

set -e

echo "============================================"
echo "Full Geometric Off-Manifold Analysis v2"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "============================================"

# Load modules
module load python/miniforge-25.3.0

# Set threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
source activate alienart

# Navigate to project
cd "$SLURM_SUBMIT_DIR"

# Create logs and output directories
mkdir -p logs
mkdir -p geometric_results_full

# Verify required files exist
echo ""
echo "Checking for pre-extracted embeddings..."
MISSING=0
for f in geometric_data/mapelites_emb.pkl geometric_data/cma_emb.pkl geometric_data/random_emb.pkl illumination_results/ref_wikiart_dino.pkl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        MISSING=1
    else
        echo "  ✓ Found: $f"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "ERROR: Missing required files. Run extract_embeddings.py first."
    exit 1
fi

echo ""
echo "Python: $(which python)"
echo ""
echo "Running comprehensive geometric analysis..."
echo ""

# ============================================================================
# Run Full Analysis with All Metrics and Visualizations
# ============================================================================
python << 'PYTHON_SCRIPT'
"""
Full Geometric Off-Manifold Analysis
Implements all metrics from geometric_analysis.py with pre-extracted embeddings
"""

import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    pca_variance_threshold: float = 0.95
    n_low_variance_pcs: int = 10
    k_neighbors: int = 10
    output_dir: Path = Path("geometric_results_full")

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Embeddings
# ============================================================================
print("=" * 70)
print("LOADING EMBEDDINGS")
print("=" * 70)

# WikiArt reference
with open("illumination_results/ref_wikiart_dino.pkl", "rb") as f:
    ref_data = pickle.load(f)
ref_embeddings = ref_data.get("embeddings", ref_data.get("dino_embeddings"))
if hasattr(ref_embeddings, 'numpy'):
    ref_embeddings = ref_embeddings.numpy()
print(f"WikiArt Reference: {ref_embeddings.shape}")

# Search methods
methods_data = {}
for name, path in [
    ("random", "geometric_data/random_emb.pkl"),
    ("cma", "geometric_data/cma_emb.pkl"),
    ("map_elites", "geometric_data/mapelites_emb.pkl"),
]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    emb = data.get("embeddings", data)
    if hasattr(emb, 'numpy'):
        emb = emb.numpy()
    methods_data[name] = {
        "embeddings": emb,
        "metadata": data.get("metadata", []),
        "search_dir": data.get("search_dir", "")
    }
    print(f"{name}: {emb.shape}")

# ============================================================================
# Fit Reference Model
# ============================================================================
print("\n" + "=" * 70)
print("FITTING REFERENCE MODEL ON WIKIART")
print("=" * 70)

# Compute mean
mean = np.mean(ref_embeddings, axis=0)
centered_ref = ref_embeddings - mean
print(f"Mean computed, dimension: {mean.shape}")

# Fit Ledoit-Wolf covariance
print("Fitting Ledoit-Wolf covariance estimator...")
lw = LedoitWolf()
lw.fit(centered_ref)
precision_matrix = lw.precision_
print(f"Precision matrix: {precision_matrix.shape}, shrinkage: {lw.shrinkage_:.4f}")

# Fit full PCA
print("Fitting PCA...")
pca_full = PCA(n_components=min(ref_embeddings.shape))
pca_full.fit(centered_ref)
eigenvalues = pca_full.explained_variance_
explained_variance_ratio = pca_full.explained_variance_ratio_

# Determine number of components for threshold
cumvar = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumvar >= config.pca_variance_threshold) + 1
print(f"PCA: {n_components} components explain {cumvar[n_components-1]*100:.1f}% variance")

# Fit k-NN model
print(f"Fitting k-NN model (k={config.k_neighbors})...")
knn_model = NearestNeighbors(n_neighbors=config.k_neighbors, metric='cosine')
knn_model.fit(ref_embeddings)
print("k-NN model fitted")

# Get low-variance PC indices
low_pc_indices = np.argsort(eigenvalues)[:config.n_low_variance_pcs]
print(f"Low-variance PCs: indices {low_pc_indices}")

# Compute baseline statistics on WikiArt for percentile calculation
print("\nComputing WikiArt baseline statistics...")
# Mahalanobis
ref_mahal = np.sqrt(np.sum((centered_ref @ precision_matrix) * centered_ref, axis=1))
# k-NN distance  
ref_knn_dist, _ = knn_model.kneighbors(ref_embeddings)
ref_knn_mean = ref_knn_dist.mean(axis=1)
# Low-PC magnitude
ref_all_pcs = pca_full.transform(centered_ref)
ref_low_pc_mag = np.linalg.norm(ref_all_pcs[:, low_pc_indices], axis=1)

print(f"WikiArt Mahalanobis: mean={np.mean(ref_mahal):.3f}, std={np.std(ref_mahal):.3f}")
print(f"WikiArt k-NN dist: mean={np.mean(ref_knn_mean):.4f}, std={np.std(ref_knn_mean):.4f}")
print(f"WikiArt Low-PC mag: mean={np.mean(ref_low_pc_mag):.4f}, std={np.std(ref_low_pc_mag):.4f}")

# ============================================================================
# Compute Metrics for Each Method
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING METRICS FOR EACH METHOD")
print("=" * 70)

results_dict = {}

for method_name, method_data in methods_data.items():
    print(f"\n--- {method_name.upper()} ---")
    embeddings = method_data["embeddings"]
    centered = embeddings - mean
    n_samples = len(embeddings)
    
    # 1. Mahalanobis Distance
    mahal = np.sqrt(np.sum((centered @ precision_matrix) * centered, axis=1))
    mahal_percentile = np.array([stats.percentileofscore(ref_mahal, m) for m in mahal])
    print(f"  Mahalanobis: {np.mean(mahal):.3f} ± {np.std(mahal):.3f}")
    print(f"  Mahalanobis percentile: {np.mean(mahal_percentile):.1f}%")
    
    # 2. PCA Reconstruction Error (SPE)
    all_pcs = pca_full.transform(centered)
    reconstructed = pca_full.inverse_transform(
        np.hstack([all_pcs[:, :n_components], np.zeros((n_samples, ref_embeddings.shape[1] - n_components))])
    )
    spe = np.sum((centered - reconstructed) ** 2, axis=1)
    print(f"  SPE (reconstruction error): {np.mean(spe):.4f} ± {np.std(spe):.4f}")
    
    # 3. Hotelling's T² (within-subspace deviation)
    # T² = sum of squared standardized scores in first n_components
    t2 = np.sum((all_pcs[:, :n_components] ** 2) / eigenvalues[:n_components], axis=1)
    print(f"  Hotelling's T²: {np.mean(t2):.3f} ± {np.std(t2):.3f}")
    
    # 4. Low-Variance PC Magnitude
    low_pc_coords = all_pcs[:, low_pc_indices]
    low_pc_magnitude = np.linalg.norm(low_pc_coords, axis=1)
    low_pc_percentile = np.array([stats.percentileofscore(ref_low_pc_mag, m) for m in low_pc_magnitude])
    print(f"  Low-PC magnitude: {np.mean(low_pc_magnitude):.4f} ± {np.std(low_pc_magnitude):.4f}")
    print(f"  Low-PC percentile: {np.mean(low_pc_percentile):.1f}%")
    
    # 5. k-NN Distance
    knn_distances, knn_indices = knn_model.kneighbors(embeddings)
    knn_distance = knn_distances.mean(axis=1)
    knn_percentile = np.array([stats.percentileofscore(ref_knn_mean, d) for d in knn_distance])
    print(f"  k-NN distance: {np.mean(knn_distance):.4f} ± {np.std(knn_distance):.4f}")
    print(f"  k-NN percentile: {np.mean(knn_percentile):.1f}%")
    
    # 6. Combined Alien Score (geometric mean of percentiles)
    combined_score = np.power(
        mahal_percentile * low_pc_percentile * knn_percentile / 1e6, 
        1/3
    )
    print(f"  Combined alien score: {np.mean(combined_score):.3f} ± {np.std(combined_score):.3f}")
    
    # Store all metrics
    results_dict[method_name] = {
        "embeddings": embeddings,
        "metadata": method_data["metadata"],
        "metrics": {
            "mahalanobis": mahal,
            "mahalanobis_percentile": mahal_percentile,
            "reconstruction_error": spe,
            "hotelling_t2": t2,
            "low_pc_magnitude": low_pc_magnitude,
            "low_pc_percentile": low_pc_percentile,
            "low_pc_coords": low_pc_coords,
            "knn_distance": knn_distance,
            "knn_percentile": knn_percentile,
            "combined_alien_score": combined_score,
        },
        "stats": {
            "mahalanobis_mean": float(np.mean(mahal)),
            "mahalanobis_std": float(np.std(mahal)),
            "mahalanobis_percentile_mean": float(np.mean(mahal_percentile)),
            "reconstruction_error_mean": float(np.mean(spe)),
            "reconstruction_error_std": float(np.std(spe)),
            "hotelling_t2_mean": float(np.mean(t2)),
            "hotelling_t2_std": float(np.std(t2)),
            "low_pc_magnitude_mean": float(np.mean(low_pc_magnitude)),
            "low_pc_magnitude_std": float(np.std(low_pc_magnitude)),
            "low_pc_percentile_mean": float(np.mean(low_pc_percentile)),
            "knn_distance_mean": float(np.mean(knn_distance)),
            "knn_distance_std": float(np.std(knn_distance)),
            "knn_percentile_mean": float(np.mean(knn_percentile)),
            "combined_alien_score_mean": float(np.mean(combined_score)),
            "combined_alien_score_std": float(np.std(combined_score)),
        }
    }

# ============================================================================
# Try to load novelty scores from search logs
# ============================================================================
print("\n" + "=" * 70)
print("EXTRACTING NOVELTY SCORES FROM SEARCH LOGS")
print("=" * 70)

novelty_data = {}
search_dirs = {
    "random": Path("outputs/run_42691366/search_dino"),
    "cma": Path("outputs/run_42708860/cma_search"),
    "map_elites": Path("outputs/run_42708270/map_elites"),
}

for method_name, search_dir in search_dirs.items():
    # Try different log file names
    for log_name in ["search_log.json", "cma_log.json", "archive_data.json"]:
        log_path = search_dir / log_name
        if log_path.exists():
            print(f"  Found {log_path}")
            with open(log_path) as f:
                log_data = json.load(f)
            
            # Extract novelty scores
            results = log_data.get("results", log_data.get("elites", []))
            novelties = []
            for r in results:
                if "novelty" in r:
                    novelties.append(r["novelty"])
                elif "clip_novelty" in r:
                    novelties.append(r["clip_novelty"])
            
            if novelties:
                novelty_data[method_name] = np.array(novelties)
                print(f"    {method_name}: {len(novelties)} novelty scores, mean={np.mean(novelties):.4f}")
                # Add to results
                if len(novelties) == len(results_dict[method_name]["embeddings"]):
                    results_dict[method_name]["metrics"]["search_novelty"] = np.array(novelties)
            else:
                print(f"    {method_name}: No novelty scores found in log")
            break

# ============================================================================
# VISUALIZATION 1: Mahalanobis Distance Comparison
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

methods = list(results_dict.keys())
colors = {'random': 'blue', 'cma': 'orange', 'map_elites': 'green', 'wikiart': 'gray'}

print("1. Mahalanobis Distance Comparison...")
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
plt.savefig(config.output_dir / "mahalanobis_comparison.png", dpi=150)
plt.close()

# ============================================================================
# VISUALIZATION 2: PCA Analysis
# ============================================================================
print("2. PCA Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PCA-Based Off-Manifold Analysis", fontsize=14)

# 2a. Reconstruction error (SPE)
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
ax.semilogy(eigenvalues, 'b-', linewidth=1)
ax.axhline(eigenvalues[n_components], color='red', linestyle='--',
           label=f'Cutoff ({n_components} PCs, {config.pca_variance_threshold*100:.0f}% var)')
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Eigenvalue (log scale)")
ax.set_title("WikiArt Eigenvalue Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "pca_analysis.png", dpi=150)
plt.close()

# ============================================================================
# VISUALIZATION 3: k-NN Analysis  
# ============================================================================
print("3. k-NN Analysis...")
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
ax.set_title(f"k-NN Distance (k={config.k_neighbors})")
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

# 3c. Novelty correlation (if available)
ax = axes[2]
has_novelty = False
for method, data in results_dict.items():
    knn_dist = data['metrics']['knn_distance']
    if 'search_novelty' in data['metrics']:
        novelty = data['metrics']['search_novelty']
        if len(novelty) == len(knn_dist):
            ax.scatter(novelty, knn_dist, alpha=0.5, s=20, label=method,
                       color=colors.get(method, 'purple'))
            has_novelty = True
            # Compute correlation
            corr = np.corrcoef(novelty, knn_dist)[0, 1]
            print(f"    {method} correlation (novelty vs k-NN): r={corr:.3f}")

if has_novelty:
    ax.set_xlabel("Search Novelty (vs History)")
    ax.set_ylabel("k-NN Distance (vs WikiArt)")
    ax.set_title("Search Novelty vs WikiArt Distance")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No novelty scores\navailable", ha='center', va='center',
            transform=ax.transAxes, fontsize=14)
    ax.set_title("Novelty Correlation (N/A)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "knn_analysis.png", dpi=150)
plt.close()

# ============================================================================
# VISUALIZATION 4: Low-Variance PC Space
# ============================================================================
print("4. Low-Variance PC Space...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Exploration of Low-Variance PC Space (Rare Directions)", fontsize=14)

# 4a. WikiArt + search results in low-variance PCs
ax = axes[0]
# WikiArt baseline (subsample for visibility)
wikiart_all_pcs = pca_full.transform(centered_ref)
wikiart_subsample = wikiart_all_pcs[::10]  # Every 10th point
ax.scatter(wikiart_subsample[:, low_pc_indices[0]], wikiart_subsample[:, low_pc_indices[1]], 
           alpha=0.2, s=10, color='gray', label='WikiArt')

for method, data in results_dict.items():
    low_coords = data['metrics']['low_pc_coords']
    ax.scatter(low_coords[:, 0], low_coords[:, 1], alpha=0.6, s=30,
               label=method, color=colors.get(method, 'purple'))

ax.set_xlabel(f"PC {low_pc_indices[0]} (λ={eigenvalues[low_pc_indices[0]]:.6f})")
ax.set_ylabel(f"PC {low_pc_indices[1]} (λ={eigenvalues[low_pc_indices[1]]:.6f})")
ax.set_title("Two Lowest-Variance Principal Components")
ax.legend()
ax.grid(True, alpha=0.3)

# 4b. Distribution of magnitudes in low-PC space
ax = axes[1]

# WikiArt baseline
wikiart_low_pc_mags = np.linalg.norm(wikiart_all_pcs[:, low_pc_indices], axis=1)
ax.hist(wikiart_low_pc_mags, bins=30, alpha=0.3, color='gray', label='WikiArt', density=True)

for method, data in results_dict.items():
    mags = data['metrics']['low_pc_magnitude']
    ax.hist(mags, bins=30, alpha=0.5, color=colors.get(method, 'purple'), 
            label=method, density=True)

ax.set_xlabel("Magnitude in Low-Variance PC Space")
ax.set_ylabel("Density")
ax.set_title(f"Distribution in {config.n_low_variance_pcs} Lowest-Variance PCs")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "low_variance_pc_space.png", dpi=150)
plt.close()

# ============================================================================
# VISUALIZATION 5: Combined Score
# ============================================================================
print("5. Combined Alien Score...")
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
metric_names = ['mahalanobis', 'reconstruction_error', 'hotelling_t2', 'low_pc_magnitude', 'knn_distance']
metric_labels = ['Mahalanobis', 'SPE', 'T²', 'Low-PC Mag', 'k-NN Dist']
x = np.arange(len(metric_names))
width = 0.8 / len(methods)

for i, (method, data) in enumerate(results_dict.items()):
    means = [np.mean(data['metrics'][m]) for m in metric_names]
    # Normalize each metric by max across methods for visual comparison
    ax.bar(x + i * width, means, width, label=method, color=colors.get(method, 'purple'), alpha=0.7)

ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(metric_labels, rotation=15, ha='right')
ax.set_ylabel("Mean Value")
ax.set_title("Off-Manifold Metrics Summary")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(config.output_dir / "combined_score.png", dpi=150)
plt.close()

# ============================================================================
# VISUALIZATION 6: Novelty Correlation (Bonus)
# ============================================================================
print("6. Novelty Correlation Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Search Novelty vs Geometric Metrics Correlation", fontsize=14)

metric_pairs = [
    ('mahalanobis', 'Mahalanobis Distance'),
    ('reconstruction_error', 'SPE'),
    ('hotelling_t2', 'Hotelling T²'),
    ('low_pc_magnitude', 'Low-PC Magnitude'),
    ('knn_distance', 'k-NN Distance'),
    ('combined_alien_score', 'Combined Score'),
]

for idx, (metric_key, metric_name) in enumerate(metric_pairs):
    ax = axes[idx // 3, idx % 3]
    correlations = {}
    
    for method, data in results_dict.items():
        metric_vals = data['metrics'][metric_key]
        if 'search_novelty' in data['metrics']:
            novelty = data['metrics']['search_novelty']
            if len(novelty) == len(metric_vals):
                ax.scatter(novelty, metric_vals, alpha=0.5, s=20, label=method,
                           color=colors.get(method, 'purple'))
                correlations[method] = np.corrcoef(novelty, metric_vals)[0, 1]
    
    if correlations:
        corr_str = ", ".join([f"{m}: r={c:.2f}" for m, c in correlations.items()])
        ax.set_xlabel("Search Novelty")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}\n{corr_str}")
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, "No novelty data", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(metric_name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "novelty_correlation.png", dpi=150)
plt.close()

# ============================================================================
# Print Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("GEOMETRIC OFF-MANIFOLD ANALYSIS SUMMARY")
print("=" * 70)

# Create comparison table
metrics_to_compare = [
    ('mahalanobis_mean', 'Mahalanobis Distance'),
    ('mahalanobis_percentile_mean', 'Mahal. Percentile (%)'),
    ('reconstruction_error_mean', 'PCA Recon. Error (SPE)'),
    ('hotelling_t2_mean', "Hotelling's T²"),
    ('low_pc_magnitude_mean', 'Low-Variance PC Mag'),
    ('low_pc_percentile_mean', 'Low-PC Percentile (%)'),
    ('knn_distance_mean', 'k-NN Distance'),
    ('knn_percentile_mean', 'k-NN Percentile (%)'),
    ('combined_alien_score_mean', 'Combined Alien Score'),
]

print(f"\n{'Metric':<25} " + " ".join([f"{m:>15}" for m in methods]))
print("-" * (25 + 16 * len(methods)))

for metric_key, metric_name in metrics_to_compare:
    values = [results_dict[m]['stats'].get(metric_key, 0) for m in methods]
    # Mark the winner
    best_idx = np.argmax(values)
    row = f"{metric_name:<25} "
    for i, v in enumerate(values):
        if i == best_idx:
            row += f"*{v:>14.4f}"
        else:
            row += f" {v:>14.4f}"
    print(row)

print("\n* = highest (best) for off-manifold detection")

# Determine overall winner
wins = {m: 0 for m in methods}
for metric_key, _ in metrics_to_compare:
    values = [results_dict[m]['stats'].get(metric_key, 0) for m in methods]
    winner = methods[np.argmax(values)]
    wins[winner] += 1

print(f"\nWins by metric count: {wins}")
overall_winner = max(wins, key=wins.get)
print(f"Overall Winner: {overall_winner.upper()} ({wins[overall_winner]}/{len(metrics_to_compare)} metrics)")

# ============================================================================
# Statistical Tests (Mann-Whitney U)
# ============================================================================
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U)")
print("=" * 70)

from scipy.stats import mannwhitneyu

key_metrics = ['mahalanobis', 'knn_distance', 'low_pc_magnitude', 'combined_alien_score']

print(f"\n{'Comparison':<25} {'Metric':<20} {'U stat':>12} {'p-value':>12} {'Signif.':>10}")
print("-" * 80)

for metric in key_metrics:
    # MAP-Elites vs Random
    u_stat, p_val = mannwhitneyu(
        results_dict['map_elites']['metrics'][metric],
        results_dict['random']['metrics'][metric],
        alternative='greater'
    )
    signif = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{'MAP-Elites > Random':<25} {metric:<20} {u_stat:>12.0f} {p_val:>12.4e} {signif:>10}")
    
    # MAP-Elites vs CMA-ES
    u_stat, p_val = mannwhitneyu(
        results_dict['map_elites']['metrics'][metric],
        results_dict['cma']['metrics'][metric],
        alternative='greater'
    )
    signif = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{'MAP-Elites > CMA-ES':<25} {metric:<20} {u_stat:>12.0f} {p_val:>12.4e} {signif:>10}")

# ============================================================================
# Effect Sizes (Cohen's d)
# ============================================================================
print("\n" + "=" * 70)
print("EFFECT SIZES (Cohen's d)")
print("=" * 70)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

print(f"\n{'Comparison':<25} {'Metric':<20} {'Cohen d':>12} {'Interpretation':>15}")
print("-" * 75)

for metric in key_metrics:
    # MAP-Elites vs Random
    d = cohens_d(
        results_dict['map_elites']['metrics'][metric],
        results_dict['random']['metrics'][metric]
    )
    interp = 'huge' if abs(d) > 2 else 'very large' if abs(d) > 1.2 else 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
    print(f"{'MAP-Elites vs Random':<25} {metric:<20} {d:>12.3f} {interp:>15}")
    
    # MAP-Elites vs CMA-ES
    d = cohens_d(
        results_dict['map_elites']['metrics'][metric],
        results_dict['cma']['metrics'][metric]
    )
    interp = 'huge' if abs(d) > 2 else 'very large' if abs(d) > 1.2 else 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
    print(f"{'MAP-Elites vs CMA-ES':<25} {metric:<20} {d:>12.3f} {interp:>15}")

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

summary = {
    "methods": methods,
    "n_samples": {m: len(results_dict[m]["embeddings"]) for m in methods},
    "metrics": {method: data['stats'] for method, data in results_dict.items()},
    "reference": {
        "n_samples": len(ref_embeddings),
        "embedding_dim": ref_embeddings.shape[1],
        "n_pca_components": int(n_components),
        "pca_variance_explained": float(cumvar[n_components-1]),
    },
    "winner": overall_winner,
    "wins_by_metric": wins,
}

with open(config.output_dir / "geometric_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {config.output_dir}/geometric_summary.json")

# Save detailed results for further analysis
detailed_results = {
    method: {
        "stats": data["stats"],
        "n_samples": len(data["embeddings"]),
    }
    for method, data in results_dict.items()
}
with open(config.output_dir / "detailed_results.json", 'w') as f:
    json.dump(detailed_results, f, indent=2)
print(f"  Saved: {config.output_dir}/detailed_results.json")

print(f"\nVisualizations saved to {config.output_dir}/:")
for png in sorted(config.output_dir.glob("*.png")):
    print(f"  - {png.name}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
PYTHON_SCRIPT

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Full Geometric Analysis Complete"
echo "============================================"
echo "End Time: $(date)"
echo ""
echo "Results saved to: geometric_results_full/"
echo ""
echo "Output files:"
ls -la geometric_results_full/
echo ""
echo "============================================"
