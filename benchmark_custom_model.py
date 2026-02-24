#!/usr/bin/env python3
"""
Benchmark the custom-trained Rice model on TAIR10 data.
Compares predictions against ground truth annotations.
"""
import subprocess
import sys
from pathlib import Path

def run_helixer_prediction(fasta_path, model_path, output_gff):
    """Run Helixer prediction using the custom model."""
    cmd = [
        "./run_helixer_with_env.sh",
        "--fasta", str(fasta_path),
        "--model-filepath", str(model_path),
        "--gff-output-path", str(output_gff),
        "--species", "Arabidopsis_thaliana",
        "--subsequence-length", "21384"  # Same as training data
    ]
    
    print(f"Running Helixer with custom model...")
    print(f"Command: {' '.join(cmd)}")
    
    # Add helixer_env/bin to PATH and lib to LD_LIBRARY_PATH
    import os
    env = os.environ.copy()
    env['PATH'] = '/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/nfs/stak/users/minchle/miniconda3/envs/helixer_env/lib:/lib64:' + env.get('LD_LIBRARY_PATH', '')
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error running Helixer:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"Prediction complete: {output_gff}")
    return output_gff

def compare_predictions(predicted_gff, ground_truth_gff):
    """Compare predicted GFF with ground truth using benchmark_helixer.py logic."""
    # Use existing benchmark script
    cmd = [
        sys.executable,
        "benchmark_helixer.py",
        str(predicted_gff),
        str(ground_truth_gff)
    ]
    
    print(f"\nComparing predictions to ground truth...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Warning: Comparison had issues:")
        print(result.stderr)

if __name__ == "__main__":
    # Paths
    fasta = Path("/nfs/hpc/share/evo2_shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas")
    ground_truth = Path("/nfs/hpc/share/evo2_shared/datasets/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff")
    model = Path("training_data/rice/rice_model_v2_rigorous.h5")
    output = Path("tair10_custom_model_predictions.gff")
    
    # Verify files exist
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}")
        sys.exit(1)
    if not ground_truth.exists():
        print(f"Error: Ground truth GFF not found: {ground_truth}")
        sys.exit(1)
    if not model.exists():
        print(f"Error: Trained model not found: {model}")
        print("Make sure training has completed and the model was saved.")
        sys.exit(1)
    
    print("="*60)
    print("CUSTOM RICE MODEL BENCHMARK ON TAIR10")
    print("="*60)
    print(f"Model: {model}")
    print(f"Test Data: {fasta.name}")
    print(f"Ground Truth: {ground_truth.name}")
    print("="*60)
    
    # Run prediction
    run_helixer_prediction(fasta, model, output)
    
    # Compare results
    compare_predictions(output, ground_truth)
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print(f"Predictions saved to: {output}")
    print("="*60)
