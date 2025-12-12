# Looking for Alien Art: On/Off the Manifold

**TTIC 31270 Project**  
*Finding art that is alien to us as humans through systematic exploration of visual embedding space.*

## Overview

We are interested in finding out what drawings, arts, pictures, or images would look like that are alien to us as humans. We begin with the latent space of Foundation Models (FMs) as our exploration ground, assuming that their learned representations approximate the statistical manifold of images shared by humans (the *Platonic Representation Hypothesis*).

This project operationalizes "alien art" as images that are:
1. **Coherent** — valid outputs from a pretrained Stable Diffusion model (excluding trivial noise).
2. **Novel** — far from previously seen images in DINO embedding space.
3. **Low-density/Off-manifold** — distant from both search history AND a reference distribution of 81k WikiArt images.

We use DINO representations as a proxy for human-aligned visual novelty, and **MAP-Elites** to systematically illuminate low-density regions of the generative manifold.

## The Story Arc

Random Search -> CMA-ES -> MAP-Elites (Prompt Evolution)
      |              |                    |
  Collapse       Still collapse      Sustained novelty
  (wrong space)  (smart optimizer,   (right space!)
                  wrong space)             +
                                    Low-density illumination

**Key Insight**: Language is the key lever for exploring concept space. Even sophisticated optimization (CMA-ES) on generation parameters (theta = seed, cfg, steps) cannot escape a semantic basin. You need to vary the *prompt* to jump between basins.

## Method: The Complete Pipeline

### Random Search Baseline
- **Search space**: theta = (seed, cfg_scale, steps), fixed prompt
- **Result**: Mean novelty 0.268.
- **Insight**: Parameter variation only explores locally within one semantic basin.

```bash
python search.py --num_samples 1000 --embedding dino
```

### CMA-ES Baseline
- **Search space**: Same theta, but with CMA-ES optimization.
- **Result**: Mean novelty 0.364.
- **Insight**: Optimization improves scores marginally but remains trapped in the semantic basin.

```bash
python cma_search.py --iterations 100 --embedding dino
```

### Expanded Theta Search
- **Search space**: theta_ext = (seed, cfg, steps, eta, guidance_rescale, clip_skip, prompt_strength).
- **Result**: Mean novelty 0.385.
- **Insight**: Internal diffusion controls refine texture but do not alter manifold topology enough to find alien content.

```bash
python expanded_theta_v2.py --iterations 100
```

### MAP-Elites (Prompt Evolution + DINO)
- **Search space**: Prompts (via mutation) + theta.
- **Result**: Mean novelty 0.574, 95% grid coverage.
- **Why it works**: Prompt mutations allow jumps between conceptual basins. DINO embeddings reward visual novelty over semantic consistency.

```bash
python map_elites.py --iterations 5000 --grid-size 15
```

### Low-Density Illumination
- **Add reference distribution**: WikiArt embeddings or "typical" SD outputs
- **Metric**: Distance from reference = how "off-manifold" the image is
- **Result**: True "alien" images are high on BOTH axes (novelty AND reference distance)

```bash
# Build reference cloud
python illumination.py build --source wikiart --path /path/to/wikiart --output ref_art.pkl

# Analyze search results
python illumination.py analyze --search_dir outputs/map_elites --reference ref_art.pkl
```

## Novelty Metrics

### 1. History-based Novelty (Open-endedness)
```
novelty(z_t) = 1 - max_{s in archive} cos(z_t, z_s)
```
Measures: How different from everything we've generated so far.

### 2. Reference-based Distance (Illumination)
```
ref_distance(z) = 1 - max_{i} cos(z, z_i^ref)
```
Measures: How far from a reference distribution (WikiArt, typical SD outputs).

### 3. True Alien Score (Combined)
```
alien_score = novelty × ref_distance
```
High on both = truly alien (novel AND far from known art).

## Project Structure

```
alien_art/
├── search.py              # Random search baseline
├── cma_search.py          # CMA-ES baseline (the bridge!)
├── map_elites.py          # MAP-Elites with prompt evolution
├── illumination.py        # Low-density illumination analysis
├── geometric_analysis.py  # Geometric off-manifold analysis
├── quick_geometric.py     # Simplified geometric analysis
├── expanded_theta_v2.py   # Expanded θ with internal controls
├── extract_embeddings.py  # Utility to extract embeddings from logs
├── prompts.py             # Prompt bank + mutation operators
├── map_elites_demo.ipynb  # Interactive Colab demo
├── demo.py                # Pipeline validation
├── requirements.txt       # Dependencies (includes cma, scipy)
├── slurm_jobs/            # SLURM job scripts for cluster execution
└── outputs/
    ├── search/            # Random search results
    ├── cma_search/        # CMA-ES results
    ├── map_elites/        # MAP-Elites results
    │   ├── archive_grid.png
    │   ├── alien_gallery.png
    │   ├── typical_gallery.png
    │   └── alien_vs_typical.png
    ├── illumination/      # Illumination analysis
    ├── geometric/         # Geometric off-manifold analysis
    └── expanded_theta/    # Expanded θ comparison
        ├── basic/         # Basic θ results
        └── expanded/      # Expanded θ results
```

## Quick Start

### 1. Environment Setup
```bash
conda create -n alienart
conda activate alienart
pip install -r requirements.txt
```

### 2. Run the Complete Story

```bash
# Random baseline (shows collapse)
python search.py --num_samples 100 --embedding dino

# CMA-ES (shows smart optimizer still collapses)
python cma_search.py --iterations 100 --embedding dino

# MAP-Elites (shows sustained novelty!)
python map_elites.py --iterations 300 --grid-size 10

# Illumination analysis (optional, if you have WikiArt)
python illumination.py build --source wikiart --path /path/to/wikiart --output ref_art.pkl
python illumination.py analyze --search_dir outputs/map_elites --reference ref_art.pkl
```

### 3. Geometric Off-Manifold Analysis

The geometric analysis provides rigorous statistical measures of how "off-manifold" generated images are:

```bash
# Step 1: Extract embeddings from search results
python extract_embeddings.py --search-dir outputs/search --output random_emb.npy
python extract_embeddings.py --search-dir outputs/cma_search --output cma_emb.npy
python extract_embeddings.py --search-dir outputs/map_elites --output mapelites_emb.npy

# Step 2: Run geometric analysis
python quick_geometric.py \
    --wikiart wikiart_embeddings.npy \
    --random random_emb.npy \
    --cma cma_emb.npy \
    --mapelites mapelites_emb.npy \
    --output-dir outputs/geometric
```

**Geometric metrics computed:**

| Metric | What it Measures |
|--------|------------------|
| **Mahalanobis Distance** | Statistical distance from WikiArt distribution (accounts for correlations) |
| **PCA Reconstruction Error** | Distance from the PCA subspace of WikiArt |
| **Low-Variance PC Magnitude** | Activity in directions WikiArt rarely uses |
| **k-NN Distance** | Average distance to nearest WikiArt images |
| **Combined Score** | Geometric mean of all metrics |

**Key insight**: High scores on ALL metrics indicate truly off-manifold images. These are geometrically distant from the entire WikiArt distribution.

### 4. Expanded Theta Search: Internal Diffusion Controls

Test whether internal diffusion controls open new regions beyond what external parameters (seed, CFG, steps) can reach:

```bash
# Compare basic vs expanded theta
python expanded_theta_search.py --iterations 100
```

**Theta spaces compared:**

| theta_basic | theta_expanded |
|---------|------------|
| seed | seed |
| cfg | cfg_base |
| steps | steps |
| | **eta** (DDIM stochasticity) |
| | **guidance_rescale** (CFG rescaling) |
| | **clip_skip** (skip CLIP layers) |
| | **prompt_strength** (conditioning scale) |

**Hypothesis**: Internal controls provide access to embedding regions that external parameters cannot reach.

### 5. Compare Methods
```bash
python cma_search.py --compare \
    --random-log outputs/search/search_log.json \
    --cma-log outputs/cma_search/cma_log.json
```

## Results Summary

| Method | Mean Novelty | Coverage | Key Observation |
|--------|--------------|----------|-----------------|
| Random Search | 0.268 | N/A | Baseline performance. |
| CMA-ES (Basic) | 0.364 | N/A | Optimization trapped in basin. |
| CMA-ES (Expanded) | 0.385 | N/A | Internal controls add minimal gain. |
| MAP-Elites (CLIP) | 0.297 | 38.2% | Constrained by semantic manifold. |
| **MAP-Elites (DINO)** | **0.574** | **95.1%** | **Sustained novelty and exploration.** |

## Presentation Flow

1. **Problem**: "How do we find alien art systematically?"

2. **Baseline 1 (Random)**: Novelty collapses — exploring one basin

3. **Baseline 2 (CMA-ES)**: Smart optimizer, STILL collapses — proves it's not the algorithm

4. **Insight**: "Language is the key — seeds explore locally, prompts jump basins"

5. **Solution (MAP-Elites)**: Prompt evolution + DINO evaluation
   - Sustained novelty
   - Systematic coverage (85% grid filled)
   - Show alien_vs_typical.png

6. **Geometric Analysis**: Mahalanobis distance, k-NN, Low-PC magnitude prove off-manifold

7. **Expanded θ**: Do internal controls open new regions?

## Key References

1. **MAP-Elites**: Mouret & Clune, "Illuminating search spaces by mapping elites" (2015)
2. **CMA-ES**: Hansen & Ostermeier, "Adapting arbitrary normal mutation distributions" (2001)
3. **ASAL**: Kumar et al., "Automating the Search for Artificial Life with Foundation Models" (2024)
4. **Stable Evolusion**: Colton et al., "Artist Discovery with Stable Evolusion" (2023)
5. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features" (2023)
6. **Plug-and-Play Diffusion**: Tumanyan et al., "Plug-and-Play Diffusion Features" (2023)

## Authors

Avi Oberoi (aoberoi1@uchicago.edu)  
University of Chicago / TTIC

## License

MIT