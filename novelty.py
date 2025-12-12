"""
Shared Novelty Metrics for Alien Art Search
=============================================
Standardized k-NN novelty computation used across all search methods.

Paper metric: k-NN Novelty (k=10)
  novelty = mean distance to k nearest neighbors in history/reference
  Higher = more novel (further from known samples)

Optimization: Uses FAISS for efficient k-NN on large reference sets (>10K).
  - Inspired by PatchCore (amazon-science/patchcore-inspection)
  - Falls back to numpy if FAISS not available
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Union

# Optional FAISS for efficient k-NN (significantly faster for large reference sets)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# =============================================================================
# FAISS-BASED K-NN INDEX (for large reference sets)
# =============================================================================

class FaissKNNIndex:
    """
    FAISS-based k-NN index for efficient similarity search.
    Uses inner product (cosine similarity for normalized vectors).
    
    Inspired by PatchCore (amazon-science/patchcore-inspection).
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.index = None
        self.dimension = None
        self._on_gpu = False
    
    def fit(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: [N, D] normalized embeddings
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.dimension = embeddings.shape[1]
        
        # Inner product index (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                self._on_gpu = True
            except Exception:
                # Fall back to CPU
                self._on_gpu = False
        
        self.index.add(embeddings)
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            queries: [B, D] query embeddings (normalized)
            k: Number of neighbors
            
        Returns:
            similarities: [B, k] cosine similarities (higher = more similar)
            indices: [B, k] indices of nearest neighbors
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call fit() first.")
        
        queries = np.ascontiguousarray(queries.astype(np.float32))
        k = min(k, self.index.ntotal)
        
        similarities, indices = self.index.search(queries, k)
        return similarities, indices
    
    def compute_knn_distances(self, queries: np.ndarray, k: int) -> np.ndarray:
        """
        Compute mean k-NN distance (1 - similarity) for queries.
        
        Args:
            queries: [B, D] query embeddings
            k: Number of neighbors
            
        Returns:
            distances: [B] mean distance to k nearest neighbors
        """
        similarities, _ = self.search(queries, k)
        # Convert similarity to distance
        distances = 1.0 - similarities
        return distances.mean(axis=1)
    
    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal if self.index else 0


# =============================================================================
# K-NN NOVELTY (Paper Standard)
# =============================================================================

def compute_knn_novelty(
    embedding: torch.Tensor,
    history: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute k-NN novelty score for a single embedding.
    
    Args:
        embedding: [D] embedding vector
        history: [N, D] history of embeddings
        k: Number of nearest neighbors
        
    Returns:
        Mean cosine distance to k nearest neighbors (higher = more novel)
    """
    if len(history) == 0:
        return 1.0
    
    # Ensure numpy
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    if isinstance(history, torch.Tensor):
        history = history.cpu().numpy()
    
    k = min(k, len(history))
    
    # Cosine similarity -> distance
    # sim = embedding @ history.T  # [N]
    # dist = 1 - sim
    
    # Use scipy for efficient k-NN
    similarities = history @ embedding  # [N]
    distances = 1.0 - similarities
    
    # Get k smallest distances
    knn_distances = np.partition(distances, k-1)[:k]
    
    return float(np.mean(knn_distances))


def compute_knn_novelty_batch(
    embeddings: torch.Tensor,
    history: torch.Tensor,
    k: int = 10,
) -> List[float]:
    """
    Compute k-NN novelty for a batch of embeddings.
    
    Args:
        embeddings: [B, D] batch of embeddings
        history: [N, D] history of embeddings
        k: Number of nearest neighbors
        
    Returns:
        List of novelty scores
    """
    if len(history) == 0:
        return [1.0] * len(embeddings)
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(history, torch.Tensor):
        history = history.cpu().numpy()
    
    k = min(k, len(history))
    
    # Compute all similarities at once
    similarities = embeddings @ history.T  # [B, N]
    distances = 1.0 - similarities
    
    # Get k smallest for each
    novelties = []
    for i in range(len(distances)):
        knn_dist = np.partition(distances[i], k-1)[:k]
        novelties.append(float(np.mean(knn_dist)))
    
    return novelties


def compute_knn_novelty_faiss(
    embeddings: Union[torch.Tensor, np.ndarray],
    faiss_index: FaissKNNIndex,
    k: int = 10,
) -> List[float]:
    """
    Compute k-NN novelty using pre-built FAISS index.
    Much faster for large reference sets (>10K vectors).
    
    Args:
        embeddings: [B, D] embeddings to score
        faiss_index: Pre-built FAISS index
        k: Number of nearest neighbors
        
    Returns:
        List of novelty scores (distances)
    """
    if faiss_index.size == 0:
        return [1.0] * len(embeddings)
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    distances = faiss_index.compute_knn_distances(embeddings, k)
    return distances.tolist()


# =============================================================================
# REFERENCE DISTANCE (Against WikiArt)
# =============================================================================

def compute_reference_distance(
    embedding: torch.Tensor,
    reference: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute k-NN distance from reference distribution (e.g., WikiArt).
    
    Higher = further from typical art = more "alien"
    """
    return compute_knn_novelty(embedding, reference, k=k)


def compute_reference_distance_batch(
    embeddings: torch.Tensor,
    reference: torch.Tensor,
    k: int = 10,
) -> List[float]:
    """Batch version of reference distance computation."""
    return compute_knn_novelty_batch(embeddings, reference, k=k)


# =============================================================================
# COMBINED ALIEN SCORE
# =============================================================================

def compute_alien_score(
    embedding: torch.Tensor,
    history: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
    k: int = 10,
    alpha: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute combined alien score.
    
    Args:
        embedding: [D] embedding vector
        history: [N, D] search history
        reference: [M, D] reference distribution (optional)
        k: Number of nearest neighbors
        alpha: Weight for history novelty vs reference distance
        
    Returns:
        (novelty, ref_distance, combined_score)
        If no reference: (novelty, novelty, novelty)
    """
    novelty = compute_knn_novelty(embedding, history, k=k)
    
    if reference is not None and len(reference) > 0:
        ref_dist = compute_reference_distance(embedding, reference, k=k)
        combined = alpha * novelty + (1 - alpha) * ref_dist
    else:
        ref_dist = novelty
        combined = novelty
    
    return novelty, ref_dist, combined


# =============================================================================
# LEGACY COMPATIBILITY (1 - max similarity)
# =============================================================================

def compute_novelty_legacy(
    embedding: torch.Tensor,
    history: List[torch.Tensor],
) -> float:
    """
    Legacy novelty: 1 - max(cosine similarity to history)
    Kept for backward compatibility with old results.
    """
    if len(history) == 0:
        return 1.0
    
    history_matrix = torch.stack(history)
    similarities = embedding @ history_matrix.T
    max_sim = similarities.max().item()
    
    return 1.0 - max_sim


def compute_novelty_batch_legacy(
    embeddings: torch.Tensor,
    history: List[torch.Tensor],
) -> List[float]:
    """Batch version of legacy novelty computation."""
    if len(history) == 0:
        return [1.0] * embeddings.shape[0]
    
    history_matrix = torch.stack(history)
    similarities = embeddings @ history_matrix.T
    max_sims = similarities.max(dim=1).values
    
    return (1.0 - max_sims).cpu().tolist()


# =============================================================================
# INCREMENTAL HISTORY MANAGER
# =============================================================================

class NoveltyTracker:
    """
    Efficient novelty tracking with incremental updates.
    Uses k-NN novelty as the standard metric.
    
    Optionally uses FAISS for large reference sets (>10K vectors).
    """
    
    def __init__(
        self, 
        k: int = 10, 
        reference: Optional[torch.Tensor] = None,
        use_faiss: bool = True,
        faiss_threshold: int = 10000,
    ):
        """
        Args:
            k: Number of nearest neighbors
            reference: Reference embeddings (e.g., WikiArt)
            use_faiss: Whether to use FAISS for large reference sets
            faiss_threshold: Use FAISS if reference size > threshold
        """
        self.k = k
        self.history: List[torch.Tensor] = []
        self.history_tensor: Optional[torch.Tensor] = None
        self.reference = reference
        self._rebuild_threshold = 100  # Rebuild tensor every N additions
        self._additions_since_rebuild = 0
        
        # FAISS for efficient k-NN on large reference
        self.faiss_index: Optional[FaissKNNIndex] = None
        self._use_faiss = use_faiss and FAISS_AVAILABLE
        self._faiss_threshold = faiss_threshold
        
        # Build FAISS index for large reference
        if self._use_faiss and reference is not None and len(reference) > faiss_threshold:
            self._build_faiss_index(reference)
    
    def _build_faiss_index(self, reference: torch.Tensor):
        """Build FAISS index for reference embeddings."""
        try:
            ref_np = reference.cpu().numpy() if isinstance(reference, torch.Tensor) else reference
            self.faiss_index = FaissKNNIndex(use_gpu=True)
            self.faiss_index.fit(ref_np)
            print(f"[NoveltyTracker] Built FAISS index for {len(reference)} reference embeddings")
        except Exception as e:
            print(f"[NoveltyTracker] FAISS index failed, using numpy: {e}")
            self.faiss_index = None
    
    def add(self, embedding: torch.Tensor):
        """Add embedding to history."""
        self.history.append(embedding.cpu())
        self._additions_since_rebuild += 1
        
        if self._additions_since_rebuild >= self._rebuild_threshold:
            self._rebuild_tensor()
    
    def add_batch(self, embeddings: torch.Tensor):
        """Add batch of embeddings to history."""
        for emb in embeddings:
            self.history.append(emb.cpu())
        self._additions_since_rebuild += len(embeddings)
        
        if self._additions_since_rebuild >= self._rebuild_threshold:
            self._rebuild_tensor()
    
    def _rebuild_tensor(self):
        """Rebuild history tensor for efficient computation."""
        if self.history:
            self.history_tensor = torch.stack(self.history)
        self._additions_since_rebuild = 0
    
    def compute_novelty(self, embedding: torch.Tensor) -> float:
        """Compute k-NN novelty for embedding."""
        if self.history_tensor is None or self._additions_since_rebuild > 0:
            self._rebuild_tensor()
        
        if self.history_tensor is None:
            return 1.0
        
        return compute_knn_novelty(embedding, self.history_tensor, k=self.k)
    
    def compute_novelty_batch(self, embeddings: torch.Tensor) -> List[float]:
        """Compute k-NN novelty for batch of embeddings."""
        if self.history_tensor is None or self._additions_since_rebuild > 0:
            self._rebuild_tensor()
        
        if self.history_tensor is None:
            return [1.0] * len(embeddings)
        
        return compute_knn_novelty_batch(embeddings, self.history_tensor, k=self.k)
    
    def compute_reference_distance(self, embedding: torch.Tensor) -> float:
        """Compute distance from reference distribution."""
        if self.reference is None:
            return self.compute_novelty(embedding)
        
        # Use FAISS if available for large reference
        if self.faiss_index is not None:
            return compute_knn_novelty_faiss(
                embedding.unsqueeze(0) if embedding.dim() == 1 else embedding,
                self.faiss_index,
                k=self.k,
            )[0]
        
        return compute_reference_distance(embedding, self.reference, k=self.k)
    
    def compute_reference_distance_batch(self, embeddings: torch.Tensor) -> List[float]:
        """Compute distance from reference for batch of embeddings."""
        if self.reference is None:
            return self.compute_novelty_batch(embeddings)
        
        # Use FAISS if available for large reference
        if self.faiss_index is not None:
            return compute_knn_novelty_faiss(embeddings, self.faiss_index, k=self.k)
        
        return compute_reference_distance_batch(embeddings, self.reference, k=self.k)
    
    def compute_alien_score(self, embedding: torch.Tensor, alpha: float = 0.5) -> Tuple[float, float, float]:
        """Compute combined alien score (novelty, ref_distance, combined)."""
        if self.history_tensor is None or self._additions_since_rebuild > 0:
            self._rebuild_tensor()
        
        return compute_alien_score(
            embedding, 
            self.history_tensor if self.history_tensor is not None else torch.zeros(0, embedding.shape[0]),
            self.reference,
            k=self.k,
            alpha=alpha,
        )
    
    def __len__(self):
        return len(self.history)
