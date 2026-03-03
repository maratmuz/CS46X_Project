#!/usr/bin/env python3
"""
Extract Evo2 embeddings from annotation TSV rows and save a Hugging Face Dataset.

Input TSV columns expected:
  species_id, seqname, pos_1based, feature, strand, phase, sequence,
  window_start_1based, window_end_1based, contig_len
(plus optional split column)

Output dataset columns:
  species_id (string)
  seqname (string)
  pos_1based (int32)
  strand (int8; + -> 1, - -> 0)
  feature_id (int8; Intergenic:0, UTR:1, CDS:2, Intron:3)
  phase_id (int8; 0:0, 1:1, 2:2, None:3)
  embedding (float16[D])
  window_start_1based (int32)
  window_end_1based (int32)
  contig_len (int32)
  sequence_hash (string)
  split (string, optional)

By default, full sequence strings are not stored (only a SHA-256 hash).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset, Features, Sequence, Value

from evo2 import Evo2


FEATURE_TO_ID: Dict[str, int] = {
    "Intergenic": 0,
    "UTR": 1,
    "CDS": 2,
    "Intron": 3,
}

PHASE_TO_ID: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "None": 3,
}

REQUIRED_COLUMNS = [
    "species_id",
    "seqname",
    "pos_1based",
    "feature",
    "strand",
    "phase",
    "sequence",
    "window_start_1based",
    "window_end_1based",
    "contig_len",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--in_tsv", required=True, help="Input TSV from build_annotation_dataset.py")
    p.add_argument("--out_dir", required=True, help="Directory for dataset.save_to_disk output")

    p.add_argument("--model_name", default="evo2_7b_base")
    p.add_argument("--model_local_path", default=None)
    p.add_argument("--layer_name", default="blocks.26")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--split_column", default="split", help="Optional split column name in TSV")
    p.add_argument("--default_split", default=None, help="Optional constant split value for all rows")
    p.add_argument("--store_sequence", action="store_true", help="Store raw sequence column in output")

    p.add_argument("--max_rows", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists")

    return p.parse_args()


def normalize_feature(raw: str) -> int:
    if raw is None:
        raise ValueError("feature is missing")

    val = raw.strip()
    if val in FEATURE_TO_ID:
        return FEATURE_TO_ID[val]

    lower_map = {
        "intergenic": 0,
        "utr": 1,
        "cds": 2,
        "intron": 3,
    }
    key = val.lower()
    if key in lower_map:
        return lower_map[key]

    raise ValueError(f"Unknown feature value: {raw}")


def normalize_phase(raw: str) -> int:
    if raw is None:
        return PHASE_TO_ID["None"]

    val = str(raw).strip()
    if not val or val.lower() in {"none", "nan", "null"} or val == ".":
        return PHASE_TO_ID["None"]

    if val in PHASE_TO_ID:
        return PHASE_TO_ID[val]

    if val in {"0.0", "1.0", "2.0"}:
        return PHASE_TO_ID[val[0]]

    raise ValueError(f"Unknown phase value: {raw}")


def normalize_strand(raw: str) -> int:
    if raw is None:
        raise ValueError("strand is missing")

    val = str(raw).strip()
    if val == "+":
        return 1
    if val == "-":
        return 0
    if val in {"1", "+1"}:
        return 1
    if val in {"0", "-1"}:
        return 0

    raise ValueError(f"Unknown strand value: {raw}")


def ensure_output_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_tsv_in_batches(
    in_tsv: str,
    batch_size: int,
    max_rows: Optional[int],
) -> Tuple[List[str], Iterable[List[Dict[str, str]]]]:
    with open(in_tsv, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []

        missing = set(REQUIRED_COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(f"Input TSV is missing required columns: {sorted(missing)}")

        rows: List[Dict[str, str]] = []
        count = 0

        for row in reader:
            rows.append(row)
            count += 1
            if max_rows is not None and count >= max_rows:
                break

    def _gen_batches() -> Iterable[List[Dict[str, str]]]:
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    return fieldnames, _gen_batches()


def tokenize_with_padding(
    sequences: List[str],
    tokenizer,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized = [tokenizer.tokenize(seq) for seq in sequences]
    lengths = [len(toks) for toks in tokenized]

    if any(l <= 0 for l in lengths):
        raise ValueError("Encountered empty tokenized sequence.")

    max_len = max(lengths)
    batch_size = len(sequences)
    pad_id = tokenizer.pad_id

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, toks in enumerate(tokenized):
        l = len(toks)
        input_ids[i, :l] = torch.tensor(toks, dtype=torch.long)
        attention_mask[i, :l] = 1

    last_token_index = attention_mask.sum(dim=1) - 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    last_token_index = last_token_index.to(device)

    return input_ids, attention_mask, last_token_index


def maybe_to_int32(value: str, column_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid integer in column {column_name}: {value}") from exc


def build_features(embedding_dim: int, include_split: bool, store_sequence: bool) -> Features:
    f = {
        "species_id": Value("string"),
        "seqname": Value("string"),
        "pos_1based": Value("int32"),
        "strand": Value("int8"),
        "feature_id": Value("int8"),
        "phase_id": Value("int8"),
        "embedding": Sequence(feature=Value("float16"), length=embedding_dim),
        "window_start_1based": Value("int32"),
        "window_end_1based": Value("int32"),
        "contig_len": Value("int32"),
        "sequence_hash": Value("string"),
    }

    if include_split:
        f["split"] = Value("string")

    if store_sequence:
        f["sequence"] = Value("string")

    return Features(f)


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but no CUDA is available; falling back to CPU.")
        device = "cpu"

    print(f"[INFO] Loading Evo2 model: {args.model_name}")
    if args.model_local_path:
        evo2_model = Evo2(args.model_name, local_path=args.model_local_path)
    else:
        evo2_model = Evo2(args.model_name)

    # Read all rows once for simple, deterministic batching and row counts.
    fieldnames, batch_iter = read_tsv_in_batches(args.in_tsv, args.batch_size, args.max_rows)

    include_split = args.default_split is not None or (args.split_column in fieldnames)

    cols: Dict[str, List] = {
        "species_id": [],
        "seqname": [],
        "pos_1based": [],
        "strand": [],
        "feature_id": [],
        "phase_id": [],
        "embedding": [],
        "window_start_1based": [],
        "window_end_1based": [],
        "contig_len": [],
        "sequence_hash": [],
    }

    if include_split:
        cols["split"] = []

    if args.store_sequence:
        cols["sequence"] = []

    embedding_dim: Optional[int] = None
    rows_processed = 0

    for batch_rows in batch_iter:
        if not batch_rows:
            continue

        sequences: List[str] = []
        filtered_rows: List[Dict[str, str]] = []

        for row in batch_rows:
            seq = (row.get("sequence") or "").strip().upper()
            if not seq:
                continue
            sequences.append(seq)
            filtered_rows.append(row)

        if not sequences:
            continue

        # Notebook-matched approach:
        # - tokenize DNA with evo2_model.tokenizer.tokenize
        # - call model(... return_embeddings=True, layer_names=[layer])
        # - take final token embedding as representation
        input_ids, attention_mask, last_token_index = tokenize_with_padding(
            sequences,
            evo2_model.tokenizer,
            device=device,
        )

        with torch.inference_mode():
            _, embeddings = evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=[args.layer_name],
            )

        layer_embeddings = embeddings[args.layer_name]

        if layer_embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings tensor shape [B, T, D], got {tuple(layer_embeddings.shape)}"
            )

        # Use attention mask-derived last-token index to handle padding correctly.
        batch_indices = torch.arange(layer_embeddings.shape[0], device=layer_embeddings.device)
        final_token_embeddings = layer_embeddings[batch_indices, last_token_index, :]
        final_token_embeddings = final_token_embeddings.to(torch.float16).cpu().numpy()

        if embedding_dim is None:
            embedding_dim = int(final_token_embeddings.shape[1])

        for i, row in enumerate(filtered_rows):
            seq = sequences[i]
            emb = final_token_embeddings[i]

            cols["species_id"].append(str(row["species_id"]))
            cols["seqname"].append(str(row["seqname"]))
            cols["pos_1based"].append(maybe_to_int32(row["pos_1based"], "pos_1based"))
            cols["strand"].append(normalize_strand(row["strand"]))
            cols["feature_id"].append(normalize_feature(row["feature"]))
            cols["phase_id"].append(normalize_phase(row["phase"]))
            cols["embedding"].append(emb.tolist())
            cols["window_start_1based"].append(
                maybe_to_int32(row["window_start_1based"], "window_start_1based")
            )
            cols["window_end_1based"].append(
                maybe_to_int32(row["window_end_1based"], "window_end_1based")
            )
            cols["contig_len"].append(maybe_to_int32(row["contig_len"], "contig_len"))
            cols["sequence_hash"].append(hashlib.sha256(seq.encode("utf-8")).hexdigest())

            if include_split:
                if args.default_split is not None:
                    split_value = args.default_split
                else:
                    split_value = str(row.get(args.split_column, ""))
                cols["split"].append(split_value)

            if args.store_sequence:
                cols["sequence"].append(seq)

            rows_processed += 1

        if args.log_every > 0 and rows_processed % args.log_every < len(filtered_rows):
            print(f"[INFO] Processed {rows_processed} rows")

    if rows_processed == 0:
        raise ValueError("No valid rows processed from input TSV.")

    if embedding_dim is None:
        raise ValueError("Failed to infer embedding dimension.")

    features = build_features(
        embedding_dim=embedding_dim,
        include_split=include_split,
        store_sequence=args.store_sequence,
    )

    dataset = Dataset.from_dict(cols, features=features)

    ensure_output_dir(args.out_dir, overwrite=args.overwrite)
    dataset.save_to_disk(args.out_dir)

    config = {
        "input_tsv": args.in_tsv,
        "out_dir": args.out_dir,
        "model_name": args.model_name,
        "model_local_path": args.model_local_path,
        "layer": args.layer_name,
        "dtype": "float16",
        "pooling_mode": "last_token",
        "tokenization": "evo2_model.tokenizer.tokenize",
        "extraction_call": "model(input_ids, return_embeddings=True, layer_names=[layer])",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "batch_size": args.batch_size,
        "rows_processed": rows_processed,
        "embedding_dim": embedding_dim,
        "feature_map": FEATURE_TO_ID,
        "phase_map": {"0": 0, "1": 1, "2": 2, "None": 3},
        "strand_map": {"+": 1, "-": 0},
        "store_sequence": args.store_sequence,
        "include_split": include_split,
        "split_column": args.split_column,
        "default_split": args.default_split,
    }

    cfg_path = os.path.join(args.out_dir, "extraction_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Saved dataset to: {args.out_dir}")
    print(f"[INFO] Saved config to: {cfg_path}")


if __name__ == "__main__":
    main()
