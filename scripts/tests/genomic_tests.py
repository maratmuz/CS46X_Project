"""
Genomic test runner aligned with genomic_evaluator style, but focused on
context-length/taxonomy sweeps. Uses data loader helpers to avoid N-heavy prompts
and loads a fresh model per context length to keep inference buffers bounded.
"""

import argparse
import csv
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent.parent
sys.path.append(str(PROJECT_ROOT))

from evo2 import Evo2  # noqa: E402
from scripts.tests import test_data_loader as data_loader  # noqa: E402


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nucleotide_accuracy(target: str, pred: str) -> float:
    length = min(len(target), len(pred))
    if length == 0:
        return 0.0
    matches = sum(1 for a, b in zip(target[:length], pred[:length]) if a == b)
    return matches / length


class GenomicTester:
    def __init__(
        self,
        temperature: float = 0.8,
        top_k: int = 32,
        top_p: float = 0.95,
        pred_len: int = 1000,
        max_seqlen: int = 12000,
    ):
        self.default_temperature = temperature
        self.default_top_k = top_k
        self.default_top_p = top_p
        self.default_pred_len = pred_len
        self.default_max_seqlen = max_seqlen

    def run(self, config_path: Path):
        cfg = OmegaConf.load(config_path)
        run = cfg.runs.seq_context_taxonomy

        models = list(run.models)
        taxonomy_variants = list(run.taxonomy)
        default_contexts = list(run.context_lengths)
        pred_len = int(run.get("pred_len", self.default_pred_len))
        temperature = float(run.get("temperature", self.default_temperature))
        top_k = int(run.get("top_k", self.default_top_k))
        top_p = float(run.get("top_p", self.default_top_p))
        max_seqlen = int(run.get("max_seqlen", self.default_max_seqlen))
        seed = int(run.seed)
        datasets = list(run.datasets)

        setup_logging()
        set_seed(seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output/tests")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"genomic_tests_{timestamp}.csv"
        results = []

        for dataset in datasets:
            label = dataset.label
            fasta = Path(dataset.fasta)
            if not fasta.exists():
                logging.warning("Skipping dataset %s; FASTA not found: %s", label, fasta)
                continue

            ds_contexts = list(dataset.get("context_lengths", default_contexts))
            ds_min_total_len = int(dataset.get("min_total_length", 0))
            ds_max_sequences = int(dataset.get("max_sequences", 5))

            windows = self._load_windows(label, fasta, ds_contexts, ds_min_total_len, ds_max_sequences, pred_len)
            if not windows:
                logging.warning("No windows loaded for dataset %s", label)
                continue

            for model_name in models:
                for ctx_len in sorted(ds_contexts):
                    total_len = ctx_len + pred_len
                    if total_len > max_seqlen:
                        logging.warning(
                            "Skipping ctx_len %d for model %s (total %d exceeds max_seqlen %d)",
                            ctx_len,
                            model_name,
                            total_len,
                            max_seqlen,
                        )
                        continue

                    logging.info("Loading model %s for context %d", model_name, ctx_len)
                    model = Evo2(model_name)

                    for taxonomy_text in taxonomy_variants:
                        taxonomy_label = taxonomy_text if taxonomy_text else "none"
                        prepend = (taxonomy_text + "\n") if taxonomy_text else ""

                        for seq_id, start, seq in windows:
                            if len(seq) < total_len:
                                continue
                            prompt_seq = seq[:ctx_len]
                            target_seq = seq[ctx_len : ctx_len + pred_len]
                            prompt = prepend + prompt_seq

                            try:
                                with torch.inference_mode():
                                    output = model.generate(
                                        prompt_seqs=[prompt],
                                        n_tokens=pred_len,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p,
                                    )
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    logging.error(
                                        "OOM | model=%s dataset=%s seq=%s ctx=%d taxonomy=%s",
                                        model_name,
                                        label,
                                        seq_id,
                                        ctx_len,
                                        taxonomy_label,
                                    )
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    continue
                                logging.error(
                                    "Generation failed | model=%s dataset=%s seq=%s ctx=%d taxonomy=%s err=%s",
                                    model_name,
                                    label,
                                    seq_id,
                                    ctx_len,
                                    taxonomy_label,
                                    e,
                                )
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            except Exception as e:
                                logging.error(
                                    "Generation failed | model=%s dataset=%s seq=%s ctx=%d taxonomy=%s err=%s",
                                    model_name,
                                    label,
                                    seq_id,
                                    ctx_len,
                                    taxonomy_label,
                                    e,
                                )
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue

                            pred_seq = output.sequences[0]
                            acc = nucleotide_accuracy(target_seq, pred_seq)

                            results.append(
                                {
                                    "dataset": label,
                                    "sequence_id": seq_id,
                                    "start": start,
                                    "model": model_name,
                                    "taxonomy": taxonomy_label,
                                    "context_len": ctx_len,
                                    "pred_len": pred_len,
                                    "accuracy": acc,
                                    "temperature": temperature,
                                    "top_k": top_k,
                                    "top_p": top_p,
                                }
                            )

                            del output
                            gc.collect()
                            torch.cuda.empty_cache()

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

        if results:
            fieldnames = [
                "dataset",
                "sequence_id",
                "start",
                "model",
                "taxonomy",
                "context_len",
                "pred_len",
                "accuracy",
                "temperature",
                "top_k",
                "top_p",
            ]
            with out_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logging.info("Wrote %d rows to %s", len(results), out_csv)
        else:
            logging.warning("No results recorded; nothing to write.")

    def _load_windows(
        self,
        label: str,
        fasta: Path,
        context_lengths: List[int],
        min_total_len: int,
        max_sequences: int,
        pred_len: int,
    ) -> List[Tuple[str, int, str]]:
        target_len = max(context_lengths) + pred_len
        total_len = max(target_len, min_total_len)
        max_n_fraction = 0.02
        if "chrom" in label:
            return data_loader.get_chrom_windows(
                fasta_path=fasta,
                window_len=total_len,
                max_windows=max_sequences,
                max_n_fraction=max_n_fraction,
                stride=total_len,
            )
        else:
            return data_loader.get_cds_windows(
                fasta_path=fasta,
                window_len=total_len,
                max_windows=max_sequences,
                max_n_fraction=max_n_fraction,
            )


def main():
    parser = argparse.ArgumentParser(description="Genomic tests runner using Evo2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tests/seq_context_taxonomy.yaml"),
        help="Path to test configuration YAML.",
    )
    args = parser.parse_args()
    tester = GenomicTester()
    tester.run(args.config)


if __name__ == "__main__":
    main()
