"""
Extract Embeddings from Search Logs
===================================
Utility to extract DINO/CLIP embeddings from existing search results
for use with geometric analysis.

Usage:
  python extract_embeddings.py --search-dir outputs/map_elites --output map_elites_embeddings.npy
  python extract_embeddings.py --search-dir outputs/search --output random_embeddings.npy
"""

import numpy as np
import json
import pickle
from pathlib import Path
import argparse
from PIL import Image
import torch
from tqdm import tqdm


def extract_from_log(search_dir: Path, embedding_type: str = "dino") -> tuple:
    """
    Extract embeddings from search log if available.
    
    Returns:
        embeddings: np.ndarray [N, D]
        novelties: np.ndarray [N] or None
        metadata: list of dicts
    """
    search_dir = Path(search_dir)
    
    # Find log file
    possible_logs = [
        "archive_data.json",
        "search_log.json", 
        "cma_log.json",
    ]
    
    log_path = None
    for name in possible_logs:
        if (search_dir / name).exists():
            log_path = search_dir / name
            break
    
    if log_path is None:
        raise FileNotFoundError(f"No log file found in {search_dir}")
    
    print(f"Loading from {log_path}...")
    with open(log_path) as f:
        data = json.load(f)
    
    # Get results list
    results = data.get("results", data.get("elites", []))
    if isinstance(results, dict):
        # MAP-Elites stores as dict by cell
        results = list(results.values())
    
    print(f"  Found {len(results)} results")
    
    # Check if embeddings are stored
    if results and f"{embedding_type}_embedding" in results[0]:
        print(f"  Extracting {embedding_type} embeddings from log...")
        embeddings = []
        novelties = []
        
        for r in results:
            emb = r.get(f"{embedding_type}_embedding")
            if emb is not None:
                embeddings.append(np.array(emb))
                novelties.append(r.get("novelty", 0))
        
        return np.array(embeddings), np.array(novelties), results
    
    elif results and "dino_embedding" in results[0]:
        # Default to dino if specific type not found
        print(f"  Extracting dino embeddings from log...")
        embeddings = [np.array(r["dino_embedding"]) for r in results]
        novelties = [r.get("novelty", 0) for r in results]
        return np.array(embeddings), np.array(novelties), results
    
    else:
        print("  Embeddings not in log, need to re-compute from images...")
        return None, None, results


def embed_images(image_paths: list, embedding_type: str = "dino", device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    """Re-embed images with batch processing for GPU efficiency."""
    
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
    
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_indices = []
        
        for j, path in enumerate(batch_paths):
            try:
                image = Image.open(path).convert('RGB')
                batch_images.append(image)
                valid_indices.append(j)
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        try:
            if embedding_type == "dino":
                inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            else:
                # CLIP needs individual preprocessing then stack
                batch_tensors = torch.stack([processor(img) for img in batch_images]).to(device)
                with torch.no_grad():
                    emb = model.encode_image(batch_tensors)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            
            embeddings.extend(emb.cpu().numpy())
        except Exception as e:
            print(f"  Batch error, falling back to individual: {e}")
            # Fallback to individual processing
            for img in batch_images:
                try:
                    if embedding_type == "dino":
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            emb = outputs.last_hidden_state[:, 0, :]
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                    else:
                        image_tensor = processor(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model.encode_image(image_tensor)
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb.squeeze(0).cpu().numpy())
                except:
                    continue
    
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from search results")
    parser.add_argument("--search-dir", type=str, required=True,
                        help="Path to search output directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for embeddings (.npy or .pkl)")
    parser.add_argument("--embedding-type", type=str, default="dino",
                        choices=["dino", "clip"],
                        help="Type of embeddings to extract")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation even if embeddings in log")
    
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    output_path = Path(args.output)
    
    # Try to extract from log
    embeddings, novelties, metadata = extract_from_log(search_dir, args.embedding_type)
    
    if embeddings is None or args.recompute:
        # Need to re-embed
        print("Re-computing embeddings from images...")
        image_paths = []
        for r in metadata:
            img_path = r.get("image_path")
            if img_path:
                # Handle relative paths - they're relative to project root, not search_dir
                img_path = Path(img_path)
                if not img_path.is_absolute():
                    # First try as-is (relative to CWD/project root)
                    if not img_path.exists():
                        # Try relative to search_dir
                        img_path = search_dir / img_path
                if img_path.exists():
                    image_paths.append(img_path)
        
        print(f"  Found {len(image_paths)} images")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = embed_images(image_paths, args.embedding_type, device)
        novelties = np.array([r.get("novelty", 0) for r in metadata[:len(embeddings)]])
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Save
    if output_path.suffix == '.npy':
        np.save(output_path, embeddings)
        # Also save novelties
        nov_path = output_path.with_name(output_path.stem + "_novelties.npy")
        np.save(nov_path, novelties)
        print(f"Saved embeddings to {output_path}")
        print(f"Saved novelties to {nov_path}")
    else:
        data = {
            'embeddings': embeddings,
            'novelties': novelties,
            'metadata': metadata,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()