import random
import numpy as np
import torch
from genomic_evaluator import GenomicEvaluator

def main():
    # Set random seeds for reproducibility
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU

    #seq_pred_cfg_1 = 'configs/eval/custom_eval_1.yaml'
    # seq_pred_cfg_1 = 'configs/eval/presets/seq_pred_1.yaml'
    # seq_pred_cfg_1 = 'configs/eval/seq_pred_2.yaml'
    seq_pred_cfg_2 = 'configs/eval/custom_eval_2.yaml'  # Midpoint mode, varying pred_len

    evaluator = GenomicEvaluator()

    evaluator.run(seq_pred_cfg_2)

    print('\nEvaluation finished, results saved to "output/eval/results_<timestamp>".\n')

if __name__ == '__main__':
    main()
