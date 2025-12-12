#!/bin/bash
#SBATCH --job-name=geom_paper
#SBATCH --account=pi-jevans
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/geometric_paper_%j.out
#SBATCH --error=logs/geometric_paper_%j.err

# ============================================================================
# Geometric Off-Manifold Analysis for Paper Results
# ============================================================================
# Uses embeddings from paper runs:
#   - paper_random_42947846 (1000 samples)
#   - paper_cma_42949466 (84 samples)
#   - paper_mapelites_42948155 (486 samples)
# WikiArt reference: 81,444 embeddings
#
# Computes:
#   - Mahalanobis Distance (Ledoit-Wolf regularization)
#   - PCA Reconstruction Error (SPE)
#   - Hotelling's T² (within-subspace deviation)
#   - Low-Variance PC Magnitude + Percentile
#   - k-NN Density Estimation
#   - Combined Alien Score
# ============================================================================

set -e

echo "============================================"
echo "Geometric Off-Manifold Analysis (Paper)"
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

# Create directories
mkdir -p logs
mkdir -p outputs/geometric_paper

# Define paths
WIKIART_REF="/project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl"
RANDOM_EMB="outputs/paper_random_42947846/embeddings.npy"
CMA_EMB="outputs/paper_cma_42949466/embeddings.npy"
MAPELITES_EMB="outputs/paper_mapelites_42948155/embeddings.npy"

# Verify required files exist
echo ""
echo "Checking for required files..."
MISSING=0
for f in "$WIKIART_REF" "$RANDOM_EMB" "$CMA_EMB" "$MAPELITES_EMB"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        MISSING=1
    else
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  ✓ Found: $f ($SIZE)"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "ERROR: Missing required files."
    exit 1
fi

echo ""
echo "Python: $(which python)"
echo ""
echo "Running comprehensive geometric analysis..."
echo ""

# ============================================================================
# Run Full Analysis
# ============================================================================
python << 'PYTHON_SCRIPT'
"""
Geometric Off-Manifold Analysis for Paper Results
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
from scipy.stats import mannwhitneyu
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
    output_dir: Path = Path("outputs/geometric_paper")

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Embeddings
# ============================================================================
print("=" * 70)
print("LOADING EMBEDDINGS")
print("=" * 70)

# WikiArt reference (81,444 embeddings)
print("Loading WikiArt reference...")
with open("/project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl", "rb") as f:
    ref_data = pickle.load(f)
ref_embeddings = ref_data.get("embeddings", ref_data.get("dino_embeddings"))
if hasattr(ref_embeddings, 'numpy'):
    ref_embeddings = ref_embeddings.numpy()
print(f"WikiArt Reference: {ref_embeddings.shape}")

# Search methods from paper runs
methods_data = {}

# Random search (1000 samples)
print("Loading Random search embeddings...")
random_emb = np.load("outputs/paper_random_42947846/embeddings.npy")
methods_data["random"] = {
    "embeddings": random_emb,
    "search_dir": "outputs/paper_random_42947846"
}
print(f"  Random: {random_emb.shape}")

# CMA-ES (84 samples)
print("Loading CMA-ES embeddings...")
cma_emb = np.load("outputs/paper_cma_42949466/embeddings.npy")
methods_data["cma"] = {
    "embeddings": cma_emb,
    "search_dir": "outputs/paper_cma_42949466"
}
print(f"  CMA-ES: {cma_emb.shape}")

# MAP-Elites (486 samples)
print("Loading MAP-Elites embeddings...")
mapelites_emb = np.load("outputs/paper_mapelites_42948155/embeddings.npy")
methods_data["map_elites"] = {
    "embeddings": mapelites_emb,
    "search_dir": "outputs/paper_mapelites_42948155"
}
print(f"  MAP-Elites: {mapelites_emb.shape}")

# ============================================================================
# Load Novelty Scores from JSON Logs
# ============================================================================
print("\nLoading novelty scores from logs...")
for method_name, method_data in methods_data.items():
    search_dir = Path(method_data["search_dir"])
    
    # Try different log file names
    for log_name in ["search_log.json", "cma_log.json", "archive_data.json"]:
        log_path = search_dir / log_name
        if log_path.exists():
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
                method_data["novelty"] = np.array(novelties)
                print(f"  {method_name}: {len(novelties)} novelty scores, mean={np.mean(novelties):.4f}")
            break

# ============================================================================
# Fit Reference Model on WikiArt
# ============================================================================
print("\n" + "=" * 70)
print("FITTING REFERENCE MODEL ON WIKIART (81,444 samples)")
print("=" * 70)

# Compute mean
mean = np.mean(ref_embeddings, axis=0)
centered_ref = ref_embeddings - mean
print(f"Mean computed, dimension: {mean.shape}")

# Fit Ledoit-Wolf covariance (regularized for high-dimensional data)
print("Fitting Ledoit-Wolf covariance estimator...")
lw = LedoitWolf()
lw.fit(centered_ref)
precision_matrix = lw.precision_
print(f"Precision matrix: {precision_matrix.shape}, shrinkage: {lw.shrinkage_:.4f}")

# Fit full PCA
print("Fitting PCA...")
n_components_max = min(ref_embeddings.shape)
pca_full = PCA(n_components=n_components_max)
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
print(f"Low-variance PCs: indices {low_pc_indices[:5]}... (first 5)")

# ============================================================================
# Compute Baseline Statistics on WikiArt
# ============================================================================
print("\nComputing WikiArt baseline statistics...")

# Mahalanobis (subsample for speed)
print("  Computing Mahalanobis distances...")
ref_mahal = np.sqrt(np.sum((centered_ref @ precision_matrix) * centered_ref, axis=1))
print(f"  WikiArt Mahalanobis: mean={np.mean(ref_mahal):.3f}, std={np.std(ref_mahal):.3f}")

# k-NN distance (subsample)
print("  Computing k-NN distances...")
ref_knn_dist, _ = knn_model.kneighbors(ref_embeddings[::10])  # Every 10th for speed
ref_knn_mean = ref_knn_dist.mean(axis=1)
print(f"  WikiArt k-NN dist (sampled): mean={np.mean(ref_knn_mean):.4f}")

# Low-PC magnitude
print("  Computing low-PC magnitudes...")
ref_all_pcs = pca_full.transform(centered_ref)
ref_low_pc_mag = np.linalg.norm(ref_all_pcs[:, low_pc_indices], axis=1)
print(f"  WikiArt Low-PC mag: mean={np.mean(ref_low_pc_mag):.4f}")

# ============================================================================
# Compute Metrics for Each Method
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING METRICS FOR EACH METHOD")
print("=" * 70)

results_dict = {}

for method_name, method_data in methods_data.items():
    print(f"\n--- {method_name.upper()} ({len(method_data['embeddings'])} samples) ---")
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
    # Reconstruct using only top n_components
    reconstructed_pcs = np.zeros_like(all_pcs)
    reconstructed_pcs[:, :n_components] = all_pcs[:, :n_components]
    reconstructed = pca_full.inverse_transform(reconstructed_pcs)
    spe = np.sum((centered - reconstructed) ** 2, axis=1)
    print(f"  SPE (reconstruction error): {np.mean(spe):.4f} ± {np.std(spe):.4f}")
    
    # 3. Hotelling's T² (within-subspace deviation)
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
    # Use full ref_knn for percentile (recompute if needed)
    full_ref_knn_dist, _ = knn_model.kneighbors(ref_embeddings[:1000])  # Sample
    full_ref_knn_mean = full_ref_knn_dist.mean(axis=1)
    knn_percentile = np.array([stats.percentileofscore(full_ref_knn_mean, d) for d in knn_distance])
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
            "n_samples": n_samples,
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
    
    # Add novelty if available
    if "novelty" in method_data:
        novelty = method_data["novelty"]
        if len(novelty) == n_samples:
            results_dict[method_name]["metrics"]["search_novelty"] = novelty
            results_dict[method_name]["stats"]["search_novelty_mean"] = float(np.mean(novelty))
            print(f"  Search novelty: {np.mean(novelty):.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

methods = list(results_dict.keys())
colors = {'random': '#1f77b4', 'cma': '#ff7f0e', 'map_elites': '#2ca02c', 'wikiart': '#7f7f7f'}

# --------------------------------------------------------------------------
# Figure 1: Mahalanobis Distance Comparison
# --------------------------------------------------------------------------
print("1. Mahalanobis Distance Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Mahalanobis Distance: How Far from WikiArt Distribution?", fontsize=14, fontweight='bold')

# 1a. Histograms
ax = axes[0]
for method in methods:
    mahal = results_dict[method]['metrics']['mahalanobis']
    ax.hist(mahal, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(mahal):.2f})",
            color=colors[method])
ax.set_xlabel("Mahalanobis Distance", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of Mahalanobis Distances")
ax.legend()
ax.grid(True, alpha=0.3)

# 1b. Box plot
ax = axes[1]
box_data = [results_dict[m]['metrics']['mahalanobis'] for m in methods]
bp = ax.boxplot(box_data, labels=[m.replace('_', '\n') for m in methods], patch_artist=True)
for patch, method in zip(bp['boxes'], methods):
    patch.set_facecolor(colors[method])
    patch.set_alpha(0.6)
ax.set_ylabel("Mahalanobis Distance", fontsize=11)
ax.set_title("Comparison Across Methods")
ax.grid(True, alpha=0.3)

# 1c. Percentile plot
ax = axes[2]
for method in methods:
    percentiles = results_dict[method]['metrics']['mahalanobis_percentile']
    ax.hist(percentiles, bins=20, alpha=0.5, label=method, color=colors[method])
ax.axvline(95, color='red', linestyle='--', linewidth=2, label='95th percentile')
ax.set_xlabel("Percentile (vs WikiArt)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("How Extreme? (Mahalanobis Percentile)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "mahalanobis_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------
# Figure 2: PCA Analysis
# --------------------------------------------------------------------------
print("2. PCA Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PCA-Based Off-Manifold Analysis", fontsize=14, fontweight='bold')

# 2a. Reconstruction error (SPE)
ax = axes[0, 0]
for method in methods:
    spe = results_dict[method]['metrics']['reconstruction_error']
    ax.hist(spe, bins=30, alpha=0.5, label=method, color=colors[method])
ax.set_xlabel("Reconstruction Error (SPE)")
ax.set_ylabel("Count")
ax.set_title("Distance from PCA Subspace")
ax.legend()
ax.grid(True, alpha=0.3)

# 2b. Hotelling's T²
ax = axes[0, 1]
for method in methods:
    t2 = results_dict[method]['metrics']['hotelling_t2']
    ax.hist(np.log10(t2 + 1), bins=30, alpha=0.5, label=method, color=colors[method])
ax.set_xlabel("log₁₀(Hotelling's T² + 1)")
ax.set_ylabel("Count")
ax.set_title("Deviation Within PCA Subspace")
ax.legend()
ax.grid(True, alpha=0.3)

# 2c. Low-variance PC magnitude
ax = axes[0, 2]
for method in methods:
    low_mag = results_dict[method]['metrics']['low_pc_magnitude']
    ax.hist(low_mag, bins=30, alpha=0.5, label=method, color=colors[method])
ax.set_xlabel("Magnitude in Low-Variance PCs")
ax.set_ylabel("Count")
ax.set_title("Activity in Rare Directions")
ax.legend()
ax.grid(True, alpha=0.3)

# 2d. Low-variance PC percentile
ax = axes[1, 0]
for method in methods:
    pct = results_dict[method]['metrics']['low_pc_percentile']
    ax.hist(pct, bins=20, alpha=0.5, label=method, color=colors[method])
ax.axvline(95, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel("Percentile (Low-Variance PC Magnitude)")
ax.set_ylabel("Count")
ax.set_title("How Extreme in Rare Directions?")
ax.legend()
ax.grid(True, alpha=0.3)

# 2e. Scatter: SPE vs T²
ax = axes[1, 1]
for method in methods:
    spe = results_dict[method]['metrics']['reconstruction_error']
    t2 = results_dict[method]['metrics']['hotelling_t2']
    ax.scatter(np.log10(t2 + 1), np.log10(spe + 1), alpha=0.5, s=20, 
               label=method, color=colors[method])
ax.set_xlabel("log₁₀(T² + 1)")
ax.set_ylabel("log₁₀(SPE + 1)")
ax.set_title("Within-Subspace vs Orthogonal Deviation")
ax.legend()
ax.grid(True, alpha=0.3)

# 2f. Eigenvalue spectrum
ax = axes[1, 2]
ax.semilogy(eigenvalues[:200], 'b-', linewidth=1.5)
ax.axhline(eigenvalues[n_components], color='red', linestyle='--', linewidth=2,
           label=f'Cutoff ({n_components} PCs, {config.pca_variance_threshold*100:.0f}% var)')
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Eigenvalue (log scale)")
ax.set_title("WikiArt Eigenvalue Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig(config.output_dir / "pca_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------
# Figure 3: k-NN Analysis
# --------------------------------------------------------------------------
print("3. k-NN Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("k-NN Density Analysis: Distance from WikiArt Samples", fontsize=14, fontweight='bold')

# 3a. k-NN distance histogram
ax = axes[0]
for method in methods:
    knn_dist = results_dict[method]['metrics']['knn_distance']
    ax.hist(knn_dist, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(knn_dist):.3f})",
            color=colors[method])
ax.set_xlabel("Average Distance to k Nearest WikiArt Images")
ax.set_ylabel("Count")
ax.set_title(f"k-NN Distance (k={config.k_neighbors})")
ax.legend()
ax.grid(True, alpha=0.3)

# 3b. Box plot
ax = axes[1]
box_data = [results_dict[m]['metrics']['knn_distance'] for m in methods]
bp = ax.boxplot(box_data, labels=[m.replace('_', '\n') for m in methods], patch_artist=True)
for patch, method in zip(bp['boxes'], methods):
    patch.set_facecolor(colors[method])
    patch.set_alpha(0.6)
ax.set_ylabel("k-NN Distance")
ax.set_title("Distance Comparison")
ax.grid(True, alpha=0.3)

# 3c. Novelty vs k-NN correlation
ax = axes[2]
has_novelty = False
for method in methods:
    knn_dist = results_dict[method]['metrics']['knn_distance']
    if 'search_novelty' in results_dict[method]['metrics']:
        novelty = results_dict[method]['metrics']['search_novelty']
        if len(novelty) == len(knn_dist):
            ax.scatter(novelty, knn_dist, alpha=0.5, s=20, label=method, color=colors[method])
            has_novelty = True
            corr = np.corrcoef(novelty, knn_dist)[0, 1]
            print(f"    {method}: Novelty vs k-NN correlation r={corr:.3f}")

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
plt.savefig(config.output_dir / "knn_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------
# Figure 4: Low-Variance PC Space
# --------------------------------------------------------------------------
print("4. Low-Variance PC Space...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Exploration of Low-Variance PC Space (Rare Directions)", fontsize=14, fontweight='bold')

# 4a. WikiArt + search results in low-variance PCs
ax = axes[0]
# WikiArt baseline (subsample for visibility)
wikiart_subsample = ref_all_pcs[::50]  # Every 50th point
ax.scatter(wikiart_subsample[:, low_pc_indices[0]], wikiart_subsample[:, low_pc_indices[1]], 
           alpha=0.15, s=8, color='gray', label='WikiArt')

for method in methods:
    low_coords = results_dict[method]['metrics']['low_pc_coords']
    ax.scatter(low_coords[:, 0], low_coords[:, 1], alpha=0.6, s=30,
               label=method, color=colors[method])

ax.set_xlabel(f"PC {low_pc_indices[0]} (λ={eigenvalues[low_pc_indices[0]]:.2e})")
ax.set_ylabel(f"PC {low_pc_indices[1]} (λ={eigenvalues[low_pc_indices[1]]:.2e})")
ax.set_title("Two Lowest-Variance Principal Components")
ax.legend()
ax.grid(True, alpha=0.3)

# 4b. Distribution of magnitudes
ax = axes[1]
ax.hist(ref_low_pc_mag, bins=50, alpha=0.3, color='gray', label='WikiArt', density=True)
for method in methods:
    mags = results_dict[method]['metrics']['low_pc_magnitude']
    ax.hist(mags, bins=30, alpha=0.5, color=colors[method], label=method, density=True)

ax.set_xlabel("Magnitude in Low-Variance PC Space")
ax.set_ylabel("Density")
ax.set_title(f"Distribution in {config.n_low_variance_pcs} Lowest-Variance PCs")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.output_dir / "low_variance_pc_space.png", dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------
# Figure 5: Combined Score Summary
# --------------------------------------------------------------------------
print("5. Combined Score Summary...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Combined Off-Manifold Score", fontsize=14, fontweight='bold')

# 5a. Histogram of combined scores
ax = axes[0]
for method in methods:
    score = results_dict[method]['metrics']['combined_alien_score']
    ax.hist(score, bins=30, alpha=0.5, label=f"{method} (μ={np.mean(score):.3f})",
            color=colors[method])
ax.set_xlabel("Combined Alien Score")
ax.set_ylabel("Count")
ax.set_title("Geometric Mean of Off-Manifold Metrics")
ax.legend()
ax.grid(True, alpha=0.3)

# 5b. Summary bar chart
ax = axes[1]
metric_names = ['mahalanobis_mean', 'reconstruction_error_mean', 'hotelling_t2_mean', 
                'low_pc_magnitude_mean', 'knn_distance_mean']
metric_labels = ['Mahalanobis', 'SPE', 'T²', 'Low-PC\nMag', 'k-NN\nDist']

x = np.arange(len(metric_names))
width = 0.25

for i, method in enumerate(methods):
    values = [results_dict[method]['stats'][m] for m in metric_names]
    # Normalize for visual comparison
    max_vals = [max(results_dict[m]['stats'][metric] for m in methods) for metric in metric_names]
    norm_values = [v / mv if mv > 0 else 0 for v, mv in zip(values, max_vals)]
    ax.bar(x + i * width, norm_values, width, label=method, color=colors[method], alpha=0.7)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Normalized Value")
ax.set_title("Off-Manifold Metrics (Normalized)")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(config.output_dir / "combined_score.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("GEOMETRIC OFF-MANIFOLD ANALYSIS SUMMARY")
print("=" * 70)

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
    best_idx = np.argmax(values)
    row = f"{metric_name:<25} "
    for i, v in enumerate(values):
        if i == best_idx:
            row += f"*{v:>14.4f}"
        else:
            row += f" {v:>14.4f}"
    print(row)

print("\n* = highest (most off-manifold)")

# Determine winner
wins = {m: 0 for m in methods}
for metric_key, _ in metrics_to_compare:
    values = [results_dict[m]['stats'].get(metric_key, 0) for m in methods]
    winner = methods[np.argmax(values)]
    wins[winner] += 1

print(f"\nWins by metric count: {wins}")
overall_winner = max(wins, key=wins.get)
print(f"Overall Winner: {overall_winner.upper()} ({wins[overall_winner]}/{len(metrics_to_compare)} metrics)")

# ============================================================================
# Statistical Tests
# ============================================================================
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U)")
print("=" * 70)

key_metrics = ['mahalanobis', 'knn_distance', 'low_pc_magnitude', 'combined_alien_score']

print(f"\n{'Comparison':<25} {'Metric':<22} {'U stat':>12} {'p-value':>12} {'Sig.':>6}")
print("-" * 80)

for metric in key_metrics:
    # MAP-Elites vs Random
    u_stat, p_val = mannwhitneyu(
        results_dict['map_elites']['metrics'][metric],
        results_dict['random']['metrics'][metric],
        alternative='greater'
    )
    signif = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{'MAP-Elites > Random':<25} {metric:<22} {u_stat:>12.0f} {p_val:>12.2e} {signif:>6}")
    
    # MAP-Elites vs CMA-ES
    u_stat, p_val = mannwhitneyu(
        results_dict['map_elites']['metrics'][metric],
        results_dict['cma']['metrics'][metric],
        alternative='greater'
    )
    signif = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{'MAP-Elites > CMA-ES':<25} {metric:<22} {u_stat:>12.0f} {p_val:>12.2e} {signif:>6}")

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
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

print(f"\n{'Comparison':<25} {'Metric':<22} {'Cohen d':>10} {'Interpretation':>15}")
print("-" * 75)

for metric in key_metrics:
    # MAP-Elites vs Random
    d = cohens_d(
        results_dict['map_elites']['metrics'][metric],
        results_dict['random']['metrics'][metric]
    )
    interp = 'huge' if abs(d) > 2 else 'very large' if abs(d) > 1.2 else 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
    print(f"{'MAP-Elites vs Random':<25} {metric:<22} {d:>10.3f} {interp:>15}")
    
    # MAP-Elites vs CMA-ES
    d = cohens_d(
        results_dict['map_elites']['metrics'][metric],
        results_dict['cma']['metrics'][metric]
    )
    interp = 'huge' if abs(d) > 2 else 'very large' if abs(d) > 1.2 else 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
    print(f"{'MAP-Elites vs CMA-ES':<25} {metric:<22} {d:>10.3f} {interp:>15}")

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

summary = {
    "methods": methods,
    "n_samples": {m: int(results_dict[m]["stats"]["n_samples"]) for m in methods},
    "metrics": {method: results_dict[method]['stats'] for method in methods},
    "reference": {
        "n_samples": int(len(ref_embeddings)),
        "embedding_dim": int(ref_embeddings.shape[1]),
        "n_pca_components": int(n_components),
        "pca_variance_explained": float(cumvar[n_components-1]),
    },
    "winner": overall_winner,
    "wins_by_metric": wins,
}

with open(config.output_dir / "geometric_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {config.output_dir}/geometric_summary.json")

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
echo "Geometric Analysis Complete"
echo "============================================"
echo "End Time: $(date)"
echo ""
echo "Results saved to: outputs/geometric_paper/"
ls -la outputs/geometric_paper/
echo ""
echo "============================================"
