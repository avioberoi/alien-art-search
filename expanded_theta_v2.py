"""
Expanded θ Search v2: Practical Internal Controls
=================================================
A cleaner implementation focusing on controls that actually work reliably.

θ_basic = {seed, cfg, steps}
θ_expanded = {seed, cfg_base, steps, eta, guidance_rescale, clip_skip, prompt_strength}

Internal controls:
- eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)
- guidance_rescale: Rescale CFG to reduce overexposure (0=off, 0.7=recommended)
- clip_skip: Skip last N CLIP text encoder layers (affects semantic interpretation)
- prompt_strength: Scale the text conditioning (affects prompt adherence)

Usage:
  python expanded_theta_v2.py --iterations 100 --population 8 --output-dir outputs/expanded_theta
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cma

# Use shared novelty module for k-NN computation
from novelty import compute_knn_novelty, FAISS_AVAILABLE

# Diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler

# For embeddings
from transformers import AutoImageProcessor, AutoModel


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class ExpandedThetaConfig:
    """Configuration for expanded θ search."""
    
    # Model settings - consistent with cma_search.py and map_elites.py
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    dino_model_id: str = "facebook/dinov2-base"
    
    # Fixed prompts (not evolving - we're testing θ space, not prompt space)
    # Use art-style prompt for fair WikiArt comparison
    prompts: List[str] = field(default_factory=lambda: [
        "a painting of a landscape",
    ])
    
    # θ bounds - External (same as cma_search.py)
    seed_range: Tuple[int, int] = (0, 1_000_000)
    cfg_scale_range: Tuple[float, float] = (5.0, 12.0)  # Aligned with map_elites.py
    steps_options: List[int] = field(default_factory=lambda: [20, 25, 30])  # Aligned with map_elites.py
    
    # θ bounds - Internal controls (expanded θ)
    eta_range: Tuple[float, float] = (0.0, 1.0)  # DDIM stochasticity
    guidance_rescale_range: Tuple[float, float] = (0.0, 0.7)  # CFG rescaling
    clip_skip_options: List[int] = field(default_factory=lambda: [0, 1, 2])  # Skip CLIP layers
    prompt_strength_range: Tuple[float, float] = (0.7, 1.3)  # Text conditioning scale
    
    # CMA-ES settings - consistent with cma_search.py
    population_size: int = 12
    sigma_init: float = 0.3  # Initial step size
    iterations: int = 100
    
    # Survivors
    survivor_size: int = 50
    
    # Generation settings
    image_size: int = 512
    
    # Output settings
    output_dir: Path = Path("outputs/expanded_theta")
    save_images: bool = True
    save_every: int = 20
    
    # Device settings - consistent with other files
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)


# =============================================================================
# CUSTOM PIPELINE WITH INTERNAL CONTROLS
# =============================================================================

class ExpandedControlPipeline:
    """
    Wrapper around SD pipeline with expanded θ controls.
    """
    
    def __init__(self, config: ExpandedThetaConfig):
        self.config = config
        self.device = config.device
        
        print(f"Loading {config.sd_model_id}...", flush=True)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.sd_model_id,
            torch_dtype=config.dtype,
            safety_checker=None,
        ).to(config.device)
        
        # Enable memory optimizations
        if "cuda" in config.device:
            # Try xformers first (most efficient)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  Enabled xformers memory-efficient attention", flush=True)
            except Exception as e:
                print(f"  xformers not available: {e}", flush=True)
                self.pipe.enable_attention_slicing()
                print("  Enabled attention slicing", flush=True)
        
        # Load DINO - use model ID from config for consistency
        print(f"Loading DINO encoder ({config.dino_model_id})...", flush=True)
        self.dino_processor = AutoImageProcessor.from_pretrained(config.dino_model_id)
        self.dino_model = AutoModel.from_pretrained(config.dino_model_id).to(config.device)
        self.dino_model.eval()
        
        # Try to compile DINO for faster inference
        if "cuda" in config.device:
            try:
                self.dino_model = torch.compile(self.dino_model, mode="reduce-overhead")
                print("  Compiled DINO model with torch.compile", flush=True)
            except Exception as e:
                print(f"  torch.compile not available: {e}", flush=True)
        
        print("Pipeline ready!", flush=True)
    
    def _encode_prompt_with_strength(
        self, 
        prompt: str, 
        negative_prompt: str,
        strength: float = 1.0,
        clip_skip: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt with optional strength scaling and CLIP skip.
        """
        # Get text encoder and tokenizer
        text_encoder = self.pipe.text_encoder
        tokenizer = self.pipe.tokenizer
        
        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode with optional layer skipping
        if clip_skip > 0:
            # Get hidden states from intermediate layer
            outputs = text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            # Skip last N layers (use layer -(1+clip_skip))
            layer_idx = -(1 + clip_skip)
            prompt_embeds = outputs.hidden_states[layer_idx]
            # Apply final layer norm
            prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)
        else:
            prompt_embeds = text_encoder(text_input_ids)[0]
        
        # Apply strength scaling
        if strength != 1.0:
            # Scale around the unconditional embedding
            uncond_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(self.device)
            
            if clip_skip > 0:
                uncond_outputs = text_encoder(uncond_input_ids, output_hidden_states=True)
                uncond_embeds = uncond_outputs.hidden_states[layer_idx]
                uncond_embeds = text_encoder.text_model.final_layer_norm(uncond_embeds)
            else:
                uncond_embeds = text_encoder(uncond_input_ids)[0]
            
            # Scale: prompt = uncond + strength * (prompt - uncond)
            prompt_embeds = uncond_embeds + strength * (prompt_embeds - uncond_embeds)
        
        # Negative prompt
        neg_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_input_ids = neg_inputs.input_ids.to(self.device)
        
        if clip_skip > 0:
            neg_outputs = text_encoder(neg_input_ids, output_hidden_states=True)
            negative_embeds = neg_outputs.hidden_states[layer_idx]
            negative_embeds = text_encoder.text_model.final_layer_norm(negative_embeds)
        else:
            negative_embeds = text_encoder(neg_input_ids)[0]
        
        return prompt_embeds, negative_embeds
    
    def generate(
        self,
        prompt: str,
        seed: int,
        cfg: float,
        steps: int,
        eta: float = 0.0,
        guidance_rescale: float = 0.0,
        clip_skip: int = 0,
        prompt_strength: float = 1.0,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Generate image with expanded controls.
        
        Args:
            prompt: Text prompt
            seed: Random seed
            cfg: Guidance scale
            steps: Number of inference steps
            eta: DDIM stochasticity (0=deterministic)
            guidance_rescale: CFG rescaling factor (reduces overexposure)
            clip_skip: Skip last N CLIP text encoder layers
            prompt_strength: Scale text conditioning
            negative_prompt: Negative prompt
            
        Returns:
            image: Generated PIL image
            embedding: DINO embedding
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Set up DDIM scheduler (eta is passed at call time, not config time)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Get prompt embeddings with custom strength and clip skip
        if clip_skip > 0 or prompt_strength != 1.0:
            prompt_embeds, negative_embeds = self._encode_prompt_with_strength(
                prompt, negative_prompt, prompt_strength, clip_skip
            )
            
            with torch.no_grad():
                image = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    guidance_rescale=guidance_rescale,
                    eta=eta,  # Pass eta here
                    generator=generator,
                    height=self.config.image_size,
                    width=self.config.image_size,
                ).images[0]
        else:
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    guidance_rescale=guidance_rescale,
                    eta=eta,  # Pass eta here
                    generator=generator,
                    height=self.config.image_size,
                    width=self.config.image_size,
                ).images[0]
        
        # Get DINO embedding
        embedding = self._get_embedding(image)
        
        return image, embedding
    
    def _get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get normalized DINO embedding for a single image."""
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy()
    
    def get_embeddings_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Get normalized DINO embeddings for a batch of images."""
        if not images:
            return np.array([])
        inputs = self.dino_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()


# =============================================================================
# CMA-ES SEARCH
# =============================================================================

class ExpandedThetaSearch:
    """
    CMA-ES search over expanded θ space.
    
    θ = [seed, cfg, steps, eta, guidance_rescale, clip_skip, prompt_strength]
    """
    
    def __init__(
        self,
        config: ExpandedThetaConfig,
        reference_embeddings: Optional[np.ndarray] = None,
        mode: str = "expanded",  # "basic" or "expanded"
    ):
        self.config = config
        self.pipeline = ExpandedControlPipeline(config)
        self.reference_embeddings = reference_embeddings
        self.mode = mode
        
        # Archive
        self.archive: List[Dict] = []
        self.archive_embeddings: List[np.ndarray] = []
        
        # Survivors
        self.survivors: List[Dict] = []
        
        # θ dimension
        # Basic: [seed, cfg, steps_idx] = 3D
        # Expanded: [seed, cfg, steps_idx, eta, guidance_rescale, clip_skip_idx, prompt_strength] = 7D
        self.theta_dim = 3 if mode == "basic" else 7
        
        # History
        self.history = {
            'iteration': [],
            'best_score': [],
            'mean_score': [],
            'best_novelty': [],
            'mean_novelty': [],
            'best_ref_dist': [],
            'mean_ref_dist': [],
        }
    
    def _decode_theta(self, theta_raw: np.ndarray) -> Dict:
        """Convert normalized θ to actual parameters."""
        cfg = self.config
        theta = np.clip(theta_raw, 0, 1)
        
        params = {
            'seed': int(theta[0] * (cfg.seed_range[1] - cfg.seed_range[0]) + cfg.seed_range[0]),
            'cfg': theta[1] * (cfg.cfg_scale_range[1] - cfg.cfg_scale_range[0]) + cfg.cfg_scale_range[0],
            'steps': cfg.steps_options[int(theta[2] * (len(cfg.steps_options) - 0.01))],
        }
        
        if self.mode == "expanded":
            params.update({
                'eta': theta[3] * (cfg.eta_range[1] - cfg.eta_range[0]) + cfg.eta_range[0],
                'guidance_rescale': theta[4] * (cfg.guidance_rescale_range[1] - cfg.guidance_rescale_range[0]) + cfg.guidance_rescale_range[0],
                'clip_skip': cfg.clip_skip_options[int(theta[5] * (len(cfg.clip_skip_options) - 0.01))],
                'prompt_strength': theta[6] * (cfg.prompt_strength_range[1] - cfg.prompt_strength_range[0]) + cfg.prompt_strength_range[0],
            })
        else:
            # Basic mode: use defaults
            params.update({
                'eta': 0.0,
                'guidance_rescale': 0.0,
                'clip_skip': 0,
                'prompt_strength': 1.0,
            })
        
        return params
    
    def _compute_novelty(self, embedding: np.ndarray, k: int = 10) -> float:
        """Novelty as average distance to k nearest in archive. Paper standard: k=10."""
        if len(self.archive_embeddings) < k:
            return 1.0
        
        # Use shared novelty module
        archive_matrix = np.array(self.archive_embeddings)
        return compute_knn_novelty(
            torch.from_numpy(embedding),
            torch.from_numpy(archive_matrix),
            k=k
        )
    
    def _compute_ref_distance(self, embedding: np.ndarray, k: int = 10) -> float:
        """Average distance to k nearest reference images. Paper standard: k=10."""
        if self.reference_embeddings is None:
            return 1.0
        
        return compute_knn_novelty(
            torch.from_numpy(embedding),
            torch.from_numpy(self.reference_embeddings),
            k=k
        )
    
    def _evaluate(self, theta_raw: np.ndarray, prompt: str) -> Tuple[float, Dict]:
        """Evaluate θ configuration."""
        params = self._decode_theta(theta_raw)
        
        try:
            image, embedding = self.pipeline.generate(prompt=prompt, **params)
        except Exception as e:
            print(f"  Generation error: {e}")
            return 0.0, {'error': str(e)}
        
        novelty = self._compute_novelty(embedding)
        ref_dist = self._compute_ref_distance(embedding)
        combined_score = novelty * ref_dist
        
        return combined_score, {
            'theta_raw': theta_raw.tolist(),
            'params': params,
            'prompt': prompt,
            'embedding': embedding,
            'novelty': novelty,
            'ref_distance': ref_dist,
            'combined_score': combined_score,
            'image': image,
        }
    
    def _update_survivors(self, new_results: List[Dict]):
        """Keep top-K survivors by combined score."""
        valid_results = [r for r in new_results if 'combined_score' in r]
        all_candidates = self.survivors + valid_results
        all_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        self.survivors = all_candidates[:self.config.survivor_size]
    
    def run(self) -> Dict:
        """Run CMA-ES search."""
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}", flush=True)
        print(f"{'EXPANDED' if self.mode == 'expanded' else 'BASIC'} θ CMA-ES SEARCH", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"θ dimension: {self.theta_dim}", flush=True)
        print(f"Mode: {self.mode}", flush=True)
        if self.mode == "expanded":
            print(f"  Controls: seed, cfg, steps, eta, guidance_rescale, clip_skip, prompt_strength", flush=True)
        else:
            print(f"  Controls: seed, cfg, steps", flush=True)
        print(f"Prompts: {config.prompts}", flush=True)
        print(f"Iterations: {config.iterations}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Initialize CMA-ES
        x0 = np.ones(self.theta_dim) * 0.5
        es = cma.CMAEvolutionStrategy(
            x0,
            config.sigma_init,
            {'popsize': config.population_size, 'bounds': [0, 1], 'seed': 42}
        )
        
        pbar = tqdm(range(config.iterations), desc="CMA-ES")
        
        for iteration in pbar:
            population = es.ask()
            valid_population = []
            scores = []
            iteration_results = []
            
            for theta in population:
                prompt = config.prompts[len(self.archive) % len(config.prompts)]
                score, result = self._evaluate(theta, prompt)
                
                if 'error' not in result:
                    valid_population.append(theta)
                    scores.append(-score)  # Minimize negative = maximize
                    iteration_results.append(result)
                    self.archive.append(result)
                    self.archive_embeddings.append(result['embedding'])
            
            # Only tell CMA-ES about valid results
            if valid_population and scores:
                es.tell(valid_population, scores)
            
            self._update_survivors(iteration_results)
            
            # Record history
            valid = [r for r in iteration_results if 'combined_score' in r]
            if valid:
                best = max(valid, key=lambda x: x['combined_score'])
                self.history['iteration'].append(iteration)
                self.history['best_score'].append(best['combined_score'])
                self.history['mean_score'].append(np.mean([r['combined_score'] for r in valid]))
                self.history['best_novelty'].append(best['novelty'])
                self.history['mean_novelty'].append(np.mean([r['novelty'] for r in valid]))
                self.history['best_ref_dist'].append(best['ref_distance'])
                self.history['mean_ref_dist'].append(np.mean([r['ref_distance'] for r in valid]))
                
                pbar.set_postfix({
                    'score': f"{best['combined_score']:.3f}",
                    'nov': f"{best['novelty']:.3f}",
                    'ref': f"{best['ref_distance']:.3f}",
                })
            
            # Save periodically
            if iteration % config.save_every == 0 and valid and config.save_images:
                best['image'].save(output_dir / f"best_iter_{iteration:04d}.png")
        
        # Final save
        self._save_results(output_dir)
        self._create_visualizations(output_dir)
        
        print(f"\nSearch complete! Archive: {len(self.archive)}, Survivors: {len(self.survivors)}")
        if self.survivors:
            print(f"Best score: {self.survivors[0]['combined_score']:.4f}")
        
        return {'survivors': self.survivors, 'history': self.history, 'archive_size': len(self.archive)}
    
    def _save_results(self, output_dir: Path):
        """Save results."""
        # Sort archive by combined score
        sorted_archive = sorted(self.archive, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Save TOP 16 gallery (most novel)
        if sorted_archive and self.config.save_images:
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig.suptitle(f"Top 16 Most Novel Images ({self.mode.capitalize()} θ)", fontsize=14, fontweight='bold')
            
            for idx, ax in enumerate(axes.flat):
                if idx < len(sorted_archive):
                    r = sorted_archive[idx]
                    if 'image' in r:
                        ax.imshow(r['image'])
                        ax.set_title(f"N={r['novelty']:.3f}\nScore={r['combined_score']:.3f}", fontsize=9)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{self.mode}_gallery_top.png", dpi=150)
            plt.close()
        
        # Save BOTTOM 16 gallery (least novel) for comparison
        if sorted_archive and self.config.save_images:
            bottom_16 = sorted_archive[-16:][::-1]  # Least novel
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig.suptitle(f"Bottom 16 (Least Novel) Images ({self.mode.capitalize()} θ)", fontsize=14, fontweight='bold')
            
            for idx, ax in enumerate(axes.flat):
                if idx < len(bottom_16):
                    r = bottom_16[idx]
                    if 'image' in r:
                        ax.imshow(r['image'])
                        ax.set_title(f"N={r['novelty']:.3f}\nScore={r['combined_score']:.3f}", fontsize=9)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{self.mode}_gallery_bottom.png", dpi=150)
            plt.close()
        
        # Save survivor images
        if self.survivors and self.config.save_images:
            gallery_dir = output_dir / "survivors"
            gallery_dir.mkdir(exist_ok=True)
            for i, s in enumerate(self.survivors[:20]):
                s['image'].save(gallery_dir / f"survivor_{i:02d}_{s['combined_score']:.3f}.png")
        
        # Save log
        log = {
            'mode': self.mode,
            'theta_dim': self.theta_dim,
            'config': {
                'prompts': self.config.prompts,
                'iterations': self.config.iterations,
            },
            'history': self.history,
            'survivors': [
                {k: v for k, v in s.items() if k not in ['image', 'embedding']}
                for s in self.survivors
            ],
            'results': [
                {k: v for k, v in a.items() if k not in ['image', 'embedding']}
                for a in self.archive
            ],
        }
        
        with open(output_dir / "search_log.json", 'w') as f:
            json.dump(log, f, indent=2)
        
        # Save embeddings
        if self.archive_embeddings:
            np.save(output_dir / "embeddings.npy", np.array(self.archive_embeddings))
            np.save(output_dir / "novelties.npy", np.array([a['novelty'] for a in self.archive]))
    
    def _create_visualizations(self, output_dir: Path):
        """Create plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{'Expanded' if self.mode == 'expanded' else 'Basic'} θ Search Results", 
                     fontsize=14, fontweight='bold')
        
        h = self.history
        
        # Score over time
        ax = axes[0, 0]
        ax.plot(h['iteration'], h['best_score'], 'g-', label='Best', linewidth=2)
        ax.plot(h['iteration'], h['mean_score'], 'b--', label='Mean', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Combined Score')
        ax.set_title('Score Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Novelty
        ax = axes[0, 1]
        ax.plot(h['iteration'], h['best_novelty'], 'g-', label='Best', linewidth=2)
        ax.plot(h['iteration'], h['mean_novelty'], 'b--', label='Mean', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Novelty')
        ax.set_title('Novelty Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ref distance
        ax = axes[0, 2]
        ax.plot(h['iteration'], h['best_ref_dist'], 'g-', label='Best', linewidth=2)
        ax.plot(h['iteration'], h['mean_ref_dist'], 'b--', label='Mean', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Ref Distance')
        ax.set_title('WikiArt Distance Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter distribution (expanded mode only)
        ax = axes[1, 0]
        if self.mode == "expanded" and self.survivors:
            param_names = ['eta', 'guidance_rescale', 'prompt_strength']
            param_data = [[s['params'][p] for s in self.survivors] for p in param_names]
            ax.boxplot(param_data, labels=param_names)
            ax.set_title('Internal Control Distribution')
            ax.set_ylabel('Value')
        else:
            ax.text(0.5, 0.5, 'Basic mode\n(no internal controls)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Internal Controls')
        ax.grid(True, alpha=0.3)
        
        # Novelty vs Ref Distance scatter
        ax = axes[1, 1]
        if self.archive:
            nov = [a['novelty'] for a in self.archive]
            ref = [a['ref_distance'] for a in self.archive]
            scores = [a['combined_score'] for a in self.archive]
            scatter = ax.scatter(nov, ref, c=scores, cmap='viridis', alpha=0.6, s=15)
            plt.colorbar(scatter, ax=ax, label='Score')
            
            if self.survivors:
                s_nov = [s['novelty'] for s in self.survivors[:10]]
                s_ref = [s['ref_distance'] for s in self.survivors[:10]]
                ax.scatter(s_nov, s_ref, c='red', marker='*', s=100, label='Top 10')
                ax.legend()
        ax.set_xlabel('Novelty')
        ax.set_ylabel('Ref Distance')
        ax.set_title('Novelty vs WikiArt Distance')
        ax.grid(True, alpha=0.3)
        
        # Score histogram
        ax = axes[1, 2]
        if self.archive:
            scores = [a['combined_score'] for a in self.archive]
            ax.hist(scores, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(scores), color='blue', linestyle='--', 
                      label=f'Mean: {np.mean(scores):.3f}')
            if self.survivors:
                ax.axvline(self.survivors[0]['combined_score'], color='green', 
                          label=f'Best: {self.survivors[0]["combined_score"]:.3f}')
            ax.legend()
        ax.set_xlabel('Combined Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "search_results.png", dpi=150)
        plt.close()


# =============================================================================
# COMPARISON
# =============================================================================

def run_comparison(config: ExpandedThetaConfig, reference_embeddings: np.ndarray = None):
    """Run both basic and expanded searches and compare."""
    
    base_output = Path(config.output_dir)
    
    # Run basic θ search
    print("\n" + "="*60, flush=True)
    print("PHASE 1: BASIC θ SEARCH", flush=True)
    print("="*60, flush=True)
    
    basic_config = ExpandedThetaConfig(
        prompts=config.prompts,
        iterations=config.iterations,
        population_size=config.population_size,
        output_dir=base_output / "basic",
        save_images=config.save_images,
        save_every=config.save_every,
    )
    
    basic_search = ExpandedThetaSearch(basic_config, reference_embeddings, mode="basic")
    basic_results = basic_search.run()
    
    # Run expanded θ search
    print("\n" + "="*60, flush=True)
    print("PHASE 2: EXPANDED θ SEARCH", flush=True)
    print("="*60, flush=True)
    
    expanded_config = ExpandedThetaConfig(
        prompts=config.prompts,
        iterations=config.iterations,
        population_size=config.population_size,
        output_dir=base_output / "expanded",
        save_images=config.save_images,
        save_every=config.save_every,
    )
    
    expanded_search = ExpandedThetaSearch(expanded_config, reference_embeddings, mode="expanded")
    expanded_results = expanded_search.run()
    
    # Create comparison visualization
    create_comparison_plot(basic_search, expanded_search, base_output)
    
    return basic_results, expanded_results


def create_comparison_plot(basic: ExpandedThetaSearch, expanded: ExpandedThetaSearch, output_dir: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Basic θ vs Expanded θ Comparison\n(Do internal controls open new regions?)", 
                 fontsize=14, fontweight='bold')
    
    colors = {'basic': '#3498db', 'expanded': '#2ecc71'}
    
    # 1. Score over iterations
    ax = axes[0, 0]
    ax.plot(basic.history['iteration'], basic.history['best_score'], 
            color=colors['basic'], label='Basic θ', linewidth=2)
    ax.plot(expanded.history['iteration'], expanded.history['best_score'], 
            color=colors['expanded'], label='Expanded θ', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Combined Score')
    ax.set_title('Score Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final score distributions
    ax = axes[0, 1]
    basic_scores = [a['combined_score'] for a in basic.archive]
    expanded_scores = [a['combined_score'] for a in expanded.archive]
    
    ax.hist(basic_scores, bins=25, alpha=0.6, color=colors['basic'], 
            label=f'Basic (μ={np.mean(basic_scores):.3f})')
    ax.hist(expanded_scores, bins=25, alpha=0.6, color=colors['expanded'],
            label=f'Expanded (μ={np.mean(expanded_scores):.3f})')
    ax.set_xlabel('Combined Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax = axes[0, 2]
    bp = ax.boxplot([basic_scores, expanded_scores], labels=['Basic θ', 'Expanded θ'], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['basic'])
    bp['boxes'][1].set_facecolor(colors['expanded'])
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Combined Score')
    ax.set_title('Score Comparison')
    ax.grid(True, alpha=0.3)
    
    # 4. Novelty comparison
    ax = axes[1, 0]
    basic_nov = [a['novelty'] for a in basic.archive]
    expanded_nov = [a['novelty'] for a in expanded.archive]
    
    ax.hist(basic_nov, bins=25, alpha=0.6, color=colors['basic'],
            label=f'Basic (μ={np.mean(basic_nov):.3f})')
    ax.hist(expanded_nov, bins=25, alpha=0.6, color=colors['expanded'],
            label=f'Expanded (μ={np.mean(expanded_nov):.3f})')
    ax.set_xlabel('Novelty')
    ax.set_ylabel('Count')
    ax.set_title('Novelty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Ref distance comparison
    ax = axes[1, 1]
    basic_ref = [a['ref_distance'] for a in basic.archive]
    expanded_ref = [a['ref_distance'] for a in expanded.archive]
    
    ax.hist(basic_ref, bins=25, alpha=0.6, color=colors['basic'],
            label=f'Basic (μ={np.mean(basic_ref):.3f})')
    ax.hist(expanded_ref, bins=25, alpha=0.6, color=colors['expanded'],
            label=f'Expanded (μ={np.mean(expanded_ref):.3f})')
    ax.set_xlabel('Reference Distance')
    ax.set_ylabel('Count')
    ax.set_title('WikiArt Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
SUMMARY STATISTICS
{'='*40}

Basic θ (seed, cfg, steps):
  Mean Score:    {np.mean(basic_scores):.4f}
  Max Score:     {np.max(basic_scores):.4f}
  Mean Novelty:  {np.mean(basic_nov):.4f}
  Mean Ref Dist: {np.mean(basic_ref):.4f}

Expanded θ (+eta, guidance_rescale, 
            clip_skip, prompt_strength):
  Mean Score:    {np.mean(expanded_scores):.4f}
  Max Score:     {np.max(expanded_scores):.4f}
  Mean Novelty:  {np.mean(expanded_nov):.4f}
  Mean Ref Dist: {np.mean(expanded_ref):.4f}

IMPROVEMENT:
  Score:    {(np.mean(expanded_scores)/np.mean(basic_scores)-1)*100:+.1f}%
  Novelty:  {(np.mean(expanded_nov)/np.mean(basic_nov)-1)*100:+.1f}%
  Ref Dist: {(np.mean(expanded_ref)/np.mean(basic_ref)-1)*100:+.1f}%
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "basic_vs_expanded_comparison.png", dpi=150)
    plt.close()
    
    print(f"\nComparison saved to {output_dir / 'basic_vs_expanded_comparison.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Expanded θ Search")
    
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="outputs/expanded_theta")
    parser.add_argument("--reference", type=str, help="WikiArt embeddings path")
    parser.add_argument("--prompts", type=str, nargs="+")
    
    parser.add_argument("--mode", choices=["basic", "expanded", "compare"], default="compare",
                        help="Run basic only, expanded only, or compare both")
    
    args = parser.parse_args()
    
    config = ExpandedThetaConfig(
        iterations=args.iterations,
        population_size=args.population,
        output_dir=Path(args.output_dir),
    )
    
    if args.prompts:
        config.prompts = args.prompts
    
    # Load reference
    reference = None
    if args.reference:
        print(f"Loading reference from {args.reference}")
        ref_path = Path(args.reference)
        if ref_path.suffix == '.pkl':
            with open(ref_path, 'rb') as f:
                ref_data = pickle.load(f)
            if isinstance(ref_data, dict):
                reference = ref_data.get('embeddings', ref_data.get('dino_embeddings'))
            else:
                reference = ref_data
            if hasattr(reference, 'numpy'):
                reference = reference.numpy()
        else:
            reference = np.load(args.reference)
        print(f"  Shape: {reference.shape}")
    
    if args.mode == "compare":
        run_comparison(config, reference)
    else:
        search = ExpandedThetaSearch(config, reference, mode=args.mode)
        search.run()


if __name__ == "__main__":
    main()