from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from Bio.Seq import Seq

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_ROOT.parent))  # allow imports from scripts/data

from data.data_loader import GenomicDataLoader  # noqa: E402
from evo2 import Evo2  # noqa: E402


def setup_logging(log_dir: Path | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "eval.log"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def to_amino_acids(seq: str) -> str:
    """
    Translate nucleotide sequence to amino acids using BioPython.
    Returns single-letter amino acids; trailing stop symbols are stripped.
    """
    seq_clean = seq.upper().replace("U", "T")  # allow RNA input
    aa_seq = str(Seq(seq_clean).translate(to_stop=False))
    return aa_seq.rstrip("*")


def nucleotide_identity(seq_a: str, seq_b: str) -> Tuple[float, int, int]:
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0, 0, 0
    matches = sum(1 for a, b in zip(seq_a[:min_len], seq_b[:min_len]) if a == b)
    return matches / min_len, matches, min_len


def amino_identity(seq_a: str, seq_b: str) -> Tuple[float, int, int]:
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0, 0, 0
    matches = sum(1 for a, b in zip(seq_a[:min_len], seq_b[:min_len]) if a == b)
    return matches / min_len, matches, min_len


class GenomicEvaluator:
    def __init__(self):
        self.supported_models = {
            "Evo2": [
                "evo2_1b_base",
                "evo2_7b_base",
                "evo2_7b",
                "evo2_40b_base",
                "evo2_40b",
            ],
            "PlantCAD2": [
                "NoneATM",
            ],
        }
        self._data_loader = GenomicDataLoader()

    def run_config(
        self,
        config_path: Path,
        output_dir: Path | None,
        log_dir: Path | None,
        print_nt: bool,
        print_aa: bool,
        verbose_loader: bool,
    ):
        config = OmegaConf.load(config_path)
        runs = config.runs

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(output_dir) if output_dir else Path("output") / "eval" / f"results_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)

        for run_idx, run in enumerate(runs.values()):
            results: dict[str, dict[str, list[float]]] = {}
            eval_mode = getattr(run.eval, "mode", "seq_pred")
            repetitions = getattr(run.eval, "repetitions", getattr(run.eval, "repitions", 1))
            tests = run.eval.get(f"{eval_mode}_tests")
            if not tests:
                raise ValueError("No tests defined in config.")

            read_mode = getattr(run.eval, "read_mode", "unique")
            data_path = Path(run.data.path)
            data_format = run.data.format

            logging.info("Loading data from %s (%s)", data_path, data_format)
            self._data_loader.load(path=data_path, format=data_format, verbose=verbose_loader)

            for model_type in run.model_types:
                model = self._load_model(model_type)
                results[model_type] = {}

                for test in tests.values():
                    seq_len = int(test.seq_len)
                    pred_len = int(test.pred_len)
                    test_label = f"SeqL {seq_len}, PredL {pred_len}"
                    results[model_type][test_label] = []

                    offsets = [rep * (seq_len + pred_len) for rep in range(repetitions)]
                    self._prepare_samples(read_mode, repetitions)

                    for rep_idx in range(repetitions):
                        prompt, target = self._read_by_mode(read_mode, seq_len, pred_len, offsets[rep_idx])
                        output = self._generate(model, prompt, pred_len)
                        if output is None:
                            continue

                        pred_seq = output.sequences[0]
                        target_trimmed = target[: len(pred_seq)]
                        nt_acc, nt_matches, nt_total = nucleotide_identity(target_trimmed, pred_seq)
                        if print_aa:
                            aa_acc, aa_matches, aa_total = amino_identity(
                                to_amino_acids(target_trimmed), to_amino_acids(pred_seq)
                            )
                            logging.info(
                                "[%s rep %d] AA accuracy: %.2f%% (%d/%d)",
                                test_label,
                                rep_idx + 1,
                                aa_acc * 100 if aa_total else 0.0,
                                aa_matches if aa_total else 0,
                                aa_total,
                            )

                        results[model_type][test_label].append(nt_acc)

                        if print_nt:
                            logging.info("[%s rep %d] Target (nt): %s", test_label, rep_idx + 1, target_trimmed)
                            logging.info("[%s rep %d] Output (nt): %s", test_label, rep_idx + 1, pred_seq)
                        logging.info(
                            "[%s rep %d] Nucleotide accuracy: %.2f%% (%d/%d)",
                            test_label,
                            rep_idx + 1,
                            nt_acc * 100 if nt_total else 0.0,
                            nt_matches if nt_total else 0,
                            nt_total,
                        )

                        del output
                        gc.collect()
                        torch.cuda.empty_cache()

                del model
                gc.collect()
                torch.cuda.empty_cache()

            self._save_results(results, out_root, run_idx)

        logging.info('Evaluation finished. Results saved to "%s".', out_root)

    def run_single(
        self,
        model_type: str,
        data_path: Path,
        data_format: str,
        seq_len: int,
        pred_len: int,
        repetitions: int,
        mode: str,
        output_dir: Path | None,
        log_dir: Path | None,
        print_nt: bool,
        print_aa: bool,
        verbose_loader: bool,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(output_dir) if output_dir else Path("output") / "eval" / f"results_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)

        logging.info("Loading data from %s (%s)", data_path, data_format)
        self._data_loader.load(path=data_path, format=data_format, verbose=verbose_loader)

        model = self._load_model(model_type)
        results: dict[str, dict[str, list[float]]] = {model_type: {}}

        test_label = f"SeqL {seq_len}, PredL {pred_len}"
        results[model_type][test_label] = []

        offsets = [rep * (seq_len + pred_len) for rep in range(repetitions)]
        self._prepare_samples(mode, repetitions)

        for rep_idx in range(repetitions):
            prompt, target = self._read_by_mode(mode, seq_len, pred_len, offsets[rep_idx])
            output = self._generate(model, prompt, pred_len)
            if output is None:
                continue

            pred_seq = output.sequences[0]
            target_trimmed = target[: len(pred_seq)]
            nt_acc, nt_matches, nt_total = nucleotide_identity(target_trimmed, pred_seq)
            aa_acc = None
            if print_aa:
                aa_acc, aa_matches, aa_total = amino_identity(
                    to_amino_acids(target_trimmed), to_amino_acids(pred_seq)
                )
                logging.info(
                    "[rep %d] Amino-acid accuracy: %.2f%% (%d/%d)",
                    rep_idx + 1,
                    aa_acc * 100 if aa_total else 0.0,
                    aa_matches if aa_total else 0,
                    aa_total,
                )

            results[model_type][test_label].append(nt_acc)

            if print_nt:
                logging.info("[rep %d] Target (nt): %s", rep_idx + 1, target_trimmed)
                logging.info("[rep %d] Output (nt): %s", rep_idx + 1, pred_seq)
            logging.info(
                "[rep %d] Nucleotide accuracy: %.2f%% (%d/%d)",
                rep_idx + 1,
                nt_acc * 100 if nt_total else 0.0,
                nt_matches if nt_total else 0,
                nt_total,
            )

            del output
            gc.collect()
            torch.cuda.empty_cache()

        self._save_results(results, out_root, 0)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        logging.info('Evaluation finished. Results saved to "%s".', out_root)

    def _prepare_samples(self, mode: str, repetitions: int):
        if mode in {"unique", "midpoint"}:
            num_samples = min(repetitions, len(self._data_loader._data))
            self._data_loader.initialize_unique_samples(num_samples=num_samples)

    def _read_by_mode(self, mode: str, seq_len: int, pred_len: int, offset: int) -> Tuple[str, str]:
        if mode == "random":
            prompt, target = self._data_loader.read_random([seq_len, pred_len])
        elif mode == "unique":
            prompt, target = self._data_loader.read_unique_start([seq_len, pred_len])
        elif mode == "midpoint":
            prompt, target = self._data_loader.read_midpoint(pred_len=pred_len)
        elif mode == "offset":
            prompt, target = self._data_loader.read_offset(seq_len=seq_len, pred_len=pred_len, offset=offset)
        else:
            raise ValueError(f"Unknown read mode: {mode}")
        return prompt, target

    def _generate(self, model, prompt: str, pred_len: int):
        try:
            with torch.inference_mode():
                return model.generate(
                    prompt_seqs=[prompt],
                    n_tokens=pred_len,
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                )
        except Exception as e:
            logging.error("model.generate failed: %s", e)
            return None

    def _load_model(self, model_type: str):
        if not any(model_type in models for models in self.supported_models.values()):
            raise ValueError(f"Model type {model_type} was not found in the list of supported models.")

        if model_type in self.supported_models["Evo2"]:
            return Evo2(model_type)
        raise NotImplementedError("Only Evo2 models are currently supported.")

    def _save_results(self, results: dict, out_root: Path, run_idx: int):
        base = out_root / f"run_{run_idx}"
        full_dir = base / "full_results"
        full_dir.mkdir(parents=True, exist_ok=True)

        df_avg = pd.DataFrame(
            {
                model: {test: (sum(vals) / len(vals) if vals else float("nan")) for test, vals in tests.items()}
                for model, tests in results.items()
            }
        )
        df_avg.to_csv(base / "avg_results.csv", index=True)

        if not results:
            return
        all_tests = sorted({t for tests in results.values() for t in tests})

        for i, test_label in enumerate(all_tests, start=1):
            df_full = pd.DataFrame({model: pd.Series(tests.get(test_label, [])) for model, tests in results.items()})
            df_full.index.name = "rep"
            df_full.to_csv(full_dir / f"test_{i}.csv", index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run genomic model evaluation.")
    parser.add_argument("--config", type=Path, help="Path to evaluation YAML config.")
    parser.add_argument("--model", type=str, help="Model type to use for one-off runs (e.g., evo2_7b_base).")
    parser.add_argument("--data-path", type=Path, help="Path to data file for one-off runs.")
    parser.add_argument("--data-format", type=str, default="fasta", help="Data format (default: fasta).")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for prompt.")
    parser.add_argument("--pred-len", type=int, default=128, help="Prediction length.")
    parser.add_argument("--repetitions", type=int, default=4, help="Number of repetitions.")
    parser.add_argument(
        "--mode",
        type=str,
        default="unique",
        choices=["random", "unique", "midpoint", "offset"],
        help="Data read mode.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--log-dir", type=Path, help="Optional directory for log file.")
    parser.add_argument("--print-nt", action="store_true", help="Print nucleotide input/outputs.")
    parser.add_argument("--print-aa", action="store_true", help="Print amino-acid sequences and accuracy.")
    parser.add_argument("--verbose-loader", action="store_true", help="Print loader details when loading data.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()
    logging.info(
        "Cleared CUDA cache | Allocated: %.2f GB | Reserved: %.2f GB",
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


def main():
    args = parse_args()

    if not args.config and not (args.model and args.data_path):
        raise SystemExit("Provide --config or --model with --data-path for one-off runs.")

    setup_logging(args.log_dir)
    set_seed(args.seed)
    clear_cuda()

    evaluator = GenomicEvaluator()

    if args.config:
        evaluator.run_config(
            config_path=args.config,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            print_nt=args.print_nt,
            print_aa=args.print_aa,
            verbose_loader=args.verbose_loader,
        )
    else:
        evaluator.run_single(
            model_type=args.model,
            data_path=args.data_path,
            data_format=args.data_format,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            repetitions=args.repetitions,
            mode=args.mode,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            print_nt=args.print_nt,
            print_aa=args.print_aa,
            verbose_loader=args.verbose_loader,
        )


if __name__ == "__main__":
    main()
