#!/usr/bin/env python3
"""
Alien Art: Full Pipeline Runner
================================
Orchestrates the complete alien art discovery pipeline.

Usage:
  # Full pipeline (if you have WikiArt)
  python run_pipeline.py --wikiart_dir /path/to/wikiart --num_samples 200
  
  # Just the search (no art cloud comparison)
  python run_pipeline.py --num_samples 200 --skip_art_cloud
  
  # Quick test
  python run_pipeline.py --num_samples 20 --skip_art_cloud
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Alien Art Full Pipeline")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for search")
    parser.add_argument("--prompt", type=str, 
                        default="a painting of a landscape",
                        help="Base prompt for generation")
    parser.add_argument("--wikiart_dir", type=str, default=None,
                        help="Path to WikiArt dataset (optional)")
    parser.add_argument("--art_sample_size", type=int, default=3000,
                        help="Number of art images to sample for cloud")
    parser.add_argument("--skip_demo", action="store_true",
                        help="Skip the demo phase")
    parser.add_argument("--skip_art_cloud", action="store_true",
                        help="Skip art cloud analysis")
    parser.add_argument("--output_base", type=str, default="outputs",
                        help="Base output directory")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ALIEN ART DISCOVERY PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Prompt: {args.prompt}")
    print(f"WikiArt: {args.wikiart_dir or 'Not provided'}")
    
    # Phase 0: Demo (optional)
    if not args.skip_demo:
        run_command(
            [sys.executable, "demo.py"],
            "Phase 0: Pipeline Demo"
        )
    
    # Phase 1: Novelty Search
    search_output = output_dir / "search"
    run_command(
        [
            sys.executable, "search.py",
            "--num_samples", str(args.num_samples),
            "--prompt", args.prompt,
            "--output_dir", str(search_output),
        ],
        "Phase 1: Novelty Search"
    )
    
    # Phase 2B: Art Cloud Analysis (optional)
    if not args.skip_art_cloud and args.wikiart_dir:
        # Build art cloud if it doesn't exist
        art_cloud_path = Path("embeddings/art_cloud.pkl")
        
        if not art_cloud_path.exists():
            run_command(
                [
                    sys.executable, "art_cloud.py", "build",
                    "--wikiart_dir", args.wikiart_dir,
                    "--output", str(art_cloud_path),
                    "--sample_size", str(args.art_sample_size),
                ],
                "Phase 2B: Building Art Cloud"
            )
        else:
            print(f"\nUsing existing art cloud: {art_cloud_path}")
        
        # Analyze search results against art cloud
        analysis_output = output_dir / "analysis"
        run_command(
            [
                sys.executable, "art_cloud.py", "analyze",
                "--search_dir", str(search_output),
                "--art_cloud", str(art_cloud_path),
                "--output_dir", str(analysis_output),
            ],
            "Phase 2B: Art Cloud Analysis"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey outputs:")
    print(f"  - Search results: {search_output}")
    print(f"  - Novelty plot: {search_output}/novelty_curve.png")
    print(f"  - High novelty gallery: {search_output}/gallery_high_novelty.png")
    print(f"  - Low novelty gallery: {search_output}/gallery_low_novelty.png")
    
    if not args.skip_art_cloud and args.wikiart_dir:
        print(f"  - Art distance analysis: {output_dir / 'analysis'}")
        print(f"  - 2D scatter plot: {output_dir / 'analysis' / 'novelty_vs_art_distance.png'}")


if __name__ == "__main__":
    main()