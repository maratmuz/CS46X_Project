import argparse
import json
from pathlib import Path
import torch
import numpy as np
from seq2expression.Evo2.nn.expression_predictor import Evo2ExpressionPredictor
from seq2expression.utils.datasets import pgb_gene_exp
from seq2expression.utils.eval_metrics import r2, auroc, classification_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictions(predictor, dataset) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for i, sample in enumerate(dataset):
        pred = predictor(sample["sequence"]).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(sample["labels"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(dataset)}")
    return np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run",      required=True, help="path to model_name_timestamp run directory")
    parser.add_argument("--evo2",     default="evo2_7b")
    parser.add_argument("--layer",    default="blocks.28.mlp.l3")
    parser.add_argument("--brackets", type=int, default=3)
    args = parser.parse_args()

    # run_dir = Path(args.run)
    run_dir = Path('shared/evo2_gene_exp/training_runs/evo2_7b_2026-03-05_11-13-32')
    heads   = {d.name: d / "best.pt" for d in run_dir.iterdir() if d.is_dir() and (d / "best.pt").exists()}

    '''temporary hack. get faster eval results only sample 50 ...'''
    # dataset = pgb_gene_exp()

    for species, head_path in heads.items():
        '''temporary hack. get faster eval results only sample 50 ...'''
        dataset = pgb_gene_exp()[species]["test"]
        indices = np.random.choice(len(dataset), size=50, replace=False)
        dataset = dataset.select(indices)

        print(f"[{species}] running …")
        predictor      = Evo2ExpressionPredictor(head_path=head_path, evo2_model=args.evo2, layer_name=args.layer, device=DEVICE)
        y_pred, y_true = get_predictions(predictor, dataset)
        # y_pred, y_true = get_predictions(predictor, dataset[species]["test"])
        clf            = classification_metrics(y_true, y_pred, n_brackets=args.brackets)

        results = {
            "r2":               round(r2(y_true, y_pred), 4),
            "auroc":            round(auroc(y_true, y_pred), 4),
            "accuracy":         round(clf["accuracy"], 4),
            "precision":        round(clf["precision"], 4),
            "recall":           round(clf["recall"], 4),
            "f1":               round(clf["f1"], 4),
            "confusion_matrix": clf["confusion_matrix"],
            "tp":               clf["tp"],
            "fp":               clf["fp"],
            "fn":               clf["fn"],
            "brackets":         args.brackets,
            "evo2":             args.evo2,
            "layer":            args.layer,
        }

        with open(head_path.parent / f"results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{species}] saved → {head_path.parent / f'results.json'}")