import random
import gc
import numpy as np
import torch
from genomic_evaluator import GenomicEvaluator

def main():
    
    # Clear any existing CUDA memory from previous runs
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared CUDA cache before starting")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved\n")
    
    # Set random seeds for reproducibility
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU

    # Gene prediction mode (paper-accurate protein recovery evaluation)
    gene_pred_cfg = 'configs/eval/gene_pred_example.yaml'
    
    # Alternative configs:
    # seq_pred_cfg_1 = 'configs/eval/custom_eval_1.yaml'  # Sequence prediction mode
    # seq_pred_cfg_2 = 'configs/eval/custom_eval_2.yaml'  # Midpoint mode, varying pred_len

    evaluator = GenomicEvaluator()

    evaluator.run(gene_pred_cfg)

    print('\nGene prediction evaluation finished!')
    print('Results saved to "output/eval/results_<timestamp>/"')
    print('  - avg_results.csv: Average protein recovery per gene')
    print('  - avg_results_by_chromosome.csv: Average protein recovery per chromosome')
    print('  - full_results/: Detailed results for each gene\n')

if __name__ == '__main__':
    main()
