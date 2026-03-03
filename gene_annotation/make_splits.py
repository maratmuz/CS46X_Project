"""
Usage example:

python make_splits.py \
  --in_dir /nfs/hpc/share/evo2_shared/datasets/splits_v2/three_algae_w8192_evo2_blocks26 \
  --out_dir /nfs/hpc/share/evo2_shared/datasets/algae_splits \
  --strategy manual_species \
  --train_species Chlamydomonas_reinhardtii \
  --val_species Cyanidioschyzon_merolae \
  --test_species Ostreococcus_lucimarinus
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, List, Set, Tuple

from datasets import load_from_disk, DatasetDict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Path to dataset.save_to_disk output")
    p.add_argument("--out_dir", required=True, help="Where to save DatasetDict with splits")

    p.add_argument(
        "--strategy",
        choices=["random", "species", "contig", "manual_species"],
        default="species",
        help="How to split the dataset",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--val_frac", type=float, default=0.1)  # fraction of remaining train

    # Manual species split options
    p.add_argument(
        "--train_species",
        default="",
        help="Comma-separated species_ids for train, or @file.txt (one species_id per line)",
    )
    p.add_argument(
        "--val_species",
        default="",
        help="Comma-separated species_ids for val, or @file.txt (one species_id per line)",
    )
    p.add_argument(
        "--test_species",
        default="",
        help="Comma-separated species_ids for test, or @file.txt (one species_id per line)",
    )
    p.add_argument(
        "--unassigned_to_train",
        action="store_true",
        help="If set, any species not listed in train/val/test will be assigned to train. "
             "Otherwise, unassigned species cause an error.",
    )

    return p.parse_args()


def _read_list_arg(arg: str) -> List[str]:
    """
    Parses either:
      - "" (empty) -> []
      - "a,b,c" -> ["a","b","c"]
      - "@file.txt" -> lines from file (strip, ignore empties and comments '#')
    """
    arg = (arg or "").strip()
    if not arg:
        return []
    if arg.startswith("@"):
        path = arg[1:]
        if not os.path.exists(path):
            raise FileNotFoundError(f"List file not found: {path}")
        out: List[str] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(line)
        return out
    return [x.strip() for x in arg.split(",") if x.strip()]


def _validate_disjoint(a: Set[str], b: Set[str], name_a: str, name_b: str) -> None:
    overlap = a & b
    if overlap:
        raise ValueError(f"Species overlap between {name_a} and {name_b}: {sorted(overlap)}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    ds = load_from_disk(args.in_dir)
    if isinstance(ds, DatasetDict):
        print(f"[INFO] Input is already a DatasetDict with splits: {list(ds.keys())}")
        ds.save_to_disk(args.out_dir)
        print(f"[INFO] Saved to {args.out_dir}")
        return

    n = len(ds)
    if n == 0:
        raise ValueError("Empty dataset.")

    if args.strategy == "random":
        ds2 = ds.shuffle(seed=args.seed)
        splits = ds2.train_test_split(test_size=args.test_frac, seed=args.seed)
        tv = splits["train"].train_test_split(test_size=args.val_frac, seed=args.seed)
        dsd = DatasetDict(train=tv["train"], val=tv["test"], test=splits["test"])

    elif args.strategy in {"species", "contig"}:
        if args.strategy == "species":
            keys = list(set(ds["species_id"]))
            key_of = lambda ex: ex["species_id"]
        else:  # contig
            keys = list(set(f"{s}::{c}" for s, c in zip(ds["species_id"], ds["seqname"])))
            key_of = lambda ex: f"{ex['species_id']}::{ex['seqname']}"

        rng.shuffle(keys)
        n_keys = len(keys)

        n_test = max(1, int(args.test_frac * n_keys))
        n_val = max(1, int(args.val_frac * (n_keys - n_test)))

        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])

        def split_label(ex):
            k = key_of(ex)
            if k in test_keys:
                return "test"
            if k in val_keys:
                return "val"
            return "train"

        ds_labeled = ds.map(lambda ex: {"split": split_label(ex)})

        dsd = DatasetDict(
            train=ds_labeled.filter(lambda ex: ex["split"] == "train").remove_columns(["split"]),
            val=ds_labeled.filter(lambda ex: ex["split"] == "val").remove_columns(["split"]),
            test=ds_labeled.filter(lambda ex: ex["split"] == "test").remove_columns(["split"]),
        )

    else:  # manual_species
        dataset_species = set(ds["species_id"])

        train_list = _read_list_arg(args.train_species)
        val_list = _read_list_arg(args.val_species)
        test_list = _read_list_arg(args.test_species)

        train_species = set(train_list)
        val_species = set(val_list)
        test_species = set(test_list)

        # Disjointness checks
        _validate_disjoint(train_species, val_species, "train", "val")
        _validate_disjoint(train_species, test_species, "train", "test")
        _validate_disjoint(val_species, test_species, "val", "test")

        # Existence checks
        unknown = (train_species | val_species | test_species) - dataset_species
        if unknown:
            raise ValueError(f"Some provided species_ids are not in the dataset: {sorted(unknown)}")

        assigned = train_species | val_species | test_species
        unassigned = dataset_species - assigned

        if unassigned and not args.unassigned_to_train:
            raise ValueError(
                "Some species are unassigned. Either assign them explicitly or pass "
                "--unassigned_to_train. Unassigned species: "
                + ", ".join(sorted(unassigned))
            )

        if unassigned and args.unassigned_to_train:
            train_species |= unassigned

        def split_label(ex):
            sid = ex["species_id"]
            if sid in test_species:
                return "test"
            if sid in val_species:
                return "val"
            # train
            return "train"

        ds_labeled = ds.map(lambda ex: {"split": split_label(ex)})

        dsd = DatasetDict(
            train=ds_labeled.filter(lambda ex: ex["split"] == "train").remove_columns(["split"]),
            val=ds_labeled.filter(lambda ex: ex["split"] == "val").remove_columns(["split"]),
            test=ds_labeled.filter(lambda ex: ex["split"] == "test").remove_columns(["split"]),
        )

        print("[INFO] Manual species assignment:")
        print(f"  train_species ({len(train_species)}): {sorted(train_species)}")
        print(f"  val_species   ({len(val_species)}): {sorted(val_species)}")
        print(f"  test_species  ({len(test_species)}): {sorted(test_species)}")

    dsd.save_to_disk(args.out_dir)
    print(f"[INFO] Saved splits to: {args.out_dir}")
    print(f"[INFO] Split sizes: train={len(dsd['train'])}, val={len(dsd['val'])}, test={len(dsd['test'])}")


if __name__ == "__main__":
    main()