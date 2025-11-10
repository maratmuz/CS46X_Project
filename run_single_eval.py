import argparse
import random
import gc
import sys
import numpy as np
import torch
from pathlib import Path
from genomic_evaluator import GenomicEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run single model evaluation on a single file")
    parser.add_argument('--data_format', type=str, default='fasta', help='Data format (default: fasta)')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length for input')
    parser.add_argument('--pred_len', type=int, default=128, help='Prediction length')
    parser.add_argument('--repetitions', type=int, default=4, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()

    # Interactive model selection
    models = ['evo2_1b_base', 'evo2_7b_base', 'evo2_40b_base']
    print("Which model would you like to choose:")
    for i, model in enumerate(models, 1):
        print(f"{i}) {model}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if 1 <= choice <= len(models):
                model_type = models[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Interactive data file selection
    data_dir = Path("shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files")
    data_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix in ['.fasta', '.fas', '.fa']]
    print("\nWhich file would you like to run the model on:")
    for i, file in enumerate(data_files, 1):
        print(f"{i}) {file.name}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if 1 <= choice <= len(data_files):
                data_path = data_files[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(data_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Clear any existing CUDA memory from previous runs
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared CUDA cache before starting")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved\n")

    # Set random seeds for reproducibility
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU

    evaluator = GenomicEvaluator()
    out_root = evaluator.run_single(
        model_type=model_type,
        data_path=str(data_path),
        data_format=args.data_format,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        repetitions=args.repetitions
    )

    print(f'\nEvaluation finished, results saved to {out_root}\n')
    sys.exit(0)

if __name__ == '__main__':
    main()