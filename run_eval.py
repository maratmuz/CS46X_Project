from genomic_evaluator import GenomicEvaluator

def main():
    seq_pred_cfg_1 = 'configs/eval/custom_eval_1.yaml'
    # seq_pred_cfg_1 = 'configs/eval/presets/seq_pred_1.yaml'
    # seq_pred_cfg_1 = 'configs/eval/seq_pred_2.yaml'

    evaluator = GenomicEvaluator()

    evaluator.run(seq_pred_cfg_1)

    print('\nEvaluation finished, results saved to "runs/eval/<run_id>.\n')

if __name__ == '__main__':
    main()
