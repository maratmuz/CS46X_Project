import argparse
import random
import gc
import sys
import os
import numpy as np
import torch
from pathlib import Path
from evo2 import Evo2
from data_loader import read_chars
import pandas as pd
from datetime import datetime
import contextlib
import io

codon_map = {
    "UUU": "Phe", "UUC": "Phe", "UUA": "Leu", "UUG": "Leu",
    "UCU": "Ser", "UCC": "Ser", "UCA": "Ser", "UCG": "Ser",
    "UAU": "Tyr", "UAC": "Tyr", "UAA": "Stop", "UAG": "Stop",
    "UGU": "Cys", "UGC": "Cys", "UGA": "Stop", "UGG": "Trp",
    "CUU": "Leu", "CUC": "Leu", "CUA": "Leu", "CUG": "Leu",
    "CCU": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "CAU": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "CGU": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AUU": "Ile", "AUC": "Ile", "AUA": "Ile", "AUG": "Met",
    "ACU": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "AAU": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "AGU": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GUU": "Val", "GUC": "Val", "GUA": "Val", "GUG": "Val",
    "GCU": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "GAU": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "GGU": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly"
}

aa3_map = {
    "Phe": "Phe", "Leu": "Leu", "Ser": "Ser", "Tyr": "Tyr", "Stop": "Stop",
    "Cys": "Cys", "Trp": "Trp", "Pro": "Pro", "His": "His", "Gln": "Gln",
    "Arg": "Arg", "Ile": "Ile", "Met": "Met", "Thr": "Thr", "Asn": "Asn",
    "Lys": "Lys", "Val": "Val", "Ala": "Ala", "Asp": "Asp", "Glu": "Glu",
    "Gly": "Gly", "X": "Xaa"
}

def seq_to_aa(seq):
    seq = seq.upper().replace('T', 'U')
    aa_seq = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            aa = codon_map.get(codon, 'X')  # X for unknown codons
            aa_seq.append(aa)
    return ''.join(aa_seq)

def _load_model(model_type):
    supported_models = {
        'Evo2': [
            'evo2_1b_base',
            'evo2_7b_base',
            'evo2_7b',
            'evo2_40b_base',
            'evo2_40b',  
        ],
        'PlantCAD2': [
            'NoneATM',
        ]
    }

    if not any(model_type in models for models in supported_models.values()):
        raise ValueError(f'Model type {model_type} was not found in the list of supported models.')

    elif model_type in supported_models['Evo2']:
        model = Evo2(model_type)
    elif model_type in supported_models['PlantCAD2']:
        raise NotImplementedError("PlantCAD2 models are not yet supported.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def _sequence_identity(seq_a: str, seq_b: str) -> tuple[float, int, int]:
    # Assuming seq_a and seq_b are strings of 3-letter AA codes concatenated
    aa_a = [seq_a[i:i+3] for i in range(0, len(seq_a), 3)]
    aa_b = [seq_b[i:i+3] for i in range(0, len(seq_b), 3)]
    min_len = min(len(aa_a), len(aa_b))
    if min_len == 0:
        return 0.0, 0, 0

    matches = 0
    for aa1, aa2 in zip(aa_a[:min_len], aa_b[:min_len]):
        if aa1 == aa2:
            matches += 1

    accuracy = matches / min_len
    return accuracy, matches, min_len


def _nucleotide_identity(seq_a: str, seq_b: str) -> tuple[float, int, int]:
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0, 0, 0

    matches = sum(1 for a, b in zip(seq_a[:min_len], seq_b[:min_len]) if a == b)
    return matches / min_len, matches, min_len



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
    data_dir = Path("../shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files")
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

    results = {}

    try:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            model = _load_model(model_type)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_type}: {e}")
        sys.exit(1)

    results[model_type] = {}

    test_label = f'SeqL {args.seq_len}, PredL {args.pred_len}'
    results[model_type][test_label] = []

    for rep_idx in range(args.repetitions):
        offset = rep_idx * (args.seq_len + args.pred_len)
        coverage_end = offset + args.seq_len + args.pred_len
        print("\n" + "=" * 80)
        print(
            f"Repetition {rep_idx + 1}/{args.repetitions} | "
            f"Offset {offset:,}–{coverage_end:,} | "
            f"Prompt {args.seq_len} | Predict {args.pred_len}"
        )
        print(f"Source file: {data_path.name}")
        print("=" * 80)

        input_seq = read_chars(args.seq_len, str(data_path), offset=offset)
        label = read_chars(3 * args.pred_len, str(data_path), offset=offset + args.seq_len)

        try:
            with torch.inference_mode():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    output = model.generate(
                        prompt_seqs=[input_seq],
                        n_tokens=args.pred_len,
                        temperature=1.0,
                        top_k=1,
                        top_p=1.0,
                    )
        except Exception as e:
            print(f"[ERROR] model.generate failed for {model_type}, test {test_label}, rep {rep_idx}: {e}")
            continue

        pred_seq = output.sequences[0]
        label_nt = label[:len(pred_seq)] if len(label) >= len(pred_seq) else label
        pred_aa = seq_to_aa(pred_seq)
        label_aa = seq_to_aa(label_nt)
        aa_acc, aa_matches, aa_total = _sequence_identity(label_aa, pred_aa)
        nt_acc, nt_matches, nt_total = _nucleotide_identity(label_nt, pred_seq)
        results[model_type][test_label].append(aa_acc)

        # Print predicted and actual AA sequences (3-letter codes) with mismatches highlighted in red
        print(f"[rep {rep_idx + 1}] Model output (nt): {pred_seq}")
        print(f"[rep {rep_idx + 1}] Actual output (nt): {label_nt}")

        label_tokens = [label_aa[i:i+3] for i in range(0, len(label_aa), 3)]
        pred_tokens = [pred_aa[i:i+3] for i in range(0, len(pred_aa), 3)]
        highlighted_label = ''
        for idx, aa_label in enumerate(label_tokens):
            if idx < len(pred_tokens):
                aa_pred = pred_tokens[idx]
                if aa_label != aa_pred:
                    highlighted_label += f'\033[91m{aa_label}\033[0m'
                else:
                    highlighted_label += aa_label
            else:
                highlighted_label += aa_label

        print(f"[rep {rep_idx + 1}] Model output (aa): {pred_aa}")
        print(f"[rep {rep_idx + 1}] Actual output (aa): {highlighted_label}")
        if nt_total:
            print(
                f"[rep {rep_idx + 1}] Nucleotide accuracy: {nt_acc * 100:6.2f}% "
                f"({nt_matches}/{nt_total})"
            )
        else:
            print(f"[rep {rep_idx + 1}] Nucleotide accuracy: N/A")
        if aa_total:
            print(
                f"[rep {rep_idx + 1}] Amino-acid accuracy: {aa_acc * 100:6.2f}% "
                f"({aa_matches}/{aa_total})"
            )
        else:
            print(f"[rep {rep_idx + 1}] Amino-acid accuracy: N/A")
        print()

        del output, pred_seq

    # average results table
    df_avg = pd.DataFrame({
        model: {
            test: (sum(vals) / len(vals) if vals else float('nan'))
            for test, vals in tests.items()
        }
        for model, tests in results.items()
    })
    print("Average Results:")
    print(df_avg.to_string())

    # per-test tables
    if results:
        all_tests = sorted({t for tests in results.values() for t in tests})
        for i, test_label in enumerate(all_tests, start=1):
            df_full = pd.DataFrame({
                model: pd.Series(tests.get(test_label, []))
                for model, tests in results.items()
            })
            df_full.index.name = "rep"
            print(f"\nFull Results for {test_label}:")
            print(df_full.to_string())

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print('\nEvaluation finished\n')
    sys.exit(0)

if __name__ == '__main__':
    main()
