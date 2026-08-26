import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.convert_supplemental_loci_to_pgb import (
    ConversionError,
    convert_species,
    load_source,
    stratified_split,
)


def write_source(directory: Path, basename: str, category: str, count: int) -> None:
    fasta = directory / f"{basename}.fasta"
    tsv = directory / f"{basename}.tsv"
    with fasta.open("w", encoding="utf-8") as fasta_handle, tsv.open(
        "w", encoding="utf-8"
    ) as tsv_handle:
        tsv_handle.write("species\tdataset_id\tlocus_category\tseqid\n")
        for index in range(count):
            dataset_id = f"{basename}-{index}"
            fasta_handle.write(f">{dataset_id}|category={category}\n{'ACGT' * 3}\n")
            tsv_handle.write(f"Test species\t{dataset_id}\t{category}\tchr1\n")


class ConverterTests(unittest.TestCase):
    def test_load_source_emits_extractor_columns_and_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root, "nuclear_intergenic", "intergenic_locus", 3)
            frame = load_source(
                root, "nuclear_intergenic", "intergenic_locus", 0.0, expected_length=12
            )
            self.assertEqual(list(frame.columns[:3]), ["sequence", "name", "labels"])
            self.assertEqual(frame["labels"].tolist(), [[0.0], [0.0], [0.0]])
            self.assertEqual(frame["name"].tolist(), frame["dataset_id"].tolist())

    def test_stratified_split_is_complete_disjoint_and_deterministic(self):
        frame = pd.DataFrame(
            {
                "name": [f"intergenic-{i}" for i in range(20)]
                + [f"pseudogene-{i}" for i in range(10)],
                "label_name": ["intergenic_locus"] * 20 + ["annotated_pseudogene"] * 10,
            }
        )
        first = stratified_split(frame, "test_species", (0.8, 0.1, 0.1), 42)
        second = stratified_split(frame, "test_species", (0.8, 0.1, 0.1), 42)
        self.assertEqual(
            [len(first[name]) for name in ("train", "validation", "test")],
            [24, 3, 3],
        )
        self.assertEqual(
            [first[name]["name"].tolist() for name in first],
            [second[name]["name"].tolist() for name in second],
        )
        split_names = [set(first[name]["name"]) for name in first]
        self.assertFalse(split_names[0] & split_names[1])
        self.assertFalse(split_names[0] & split_names[2])
        self.assertFalse(split_names[1] & split_names[2])
        self.assertEqual(len(set.union(*split_names)), len(frame))

    def test_iupac_ambiguity_codes_are_normalized_to_n(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root, "nuclear_intergenic", "intergenic_locus", 1)
            fasta = root / "nuclear_intergenic.fasta"
            fasta.write_text(
                ">nuclear_intergenic-0|category=intergenic_locus\nACGTRYSWKMNN\n",
                encoding="utf-8",
            )
            frame = load_source(
                root, "nuclear_intergenic", "intergenic_locus", 0.0, expected_length=12
            )
            self.assertEqual(frame.loc[0, "sequence"], "ACGTNNNNNNNN")
            self.assertEqual(frame.loc[0, "normalized_ambiguous_bases"], 6)

    def test_fasta_tsv_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root, "nuclear_intergenic", "intergenic_locus", 2)
            with (root / "nuclear_intergenic.tsv").open("a", encoding="utf-8") as handle:
                handle.write("Test species\tmissing-from-fasta\tintergenic_locus\tchr1\n")
            with self.assertRaises(ConversionError):
                load_source(
                    root, "nuclear_intergenic", "intergenic_locus", 0.0, expected_length=12
                )

    def test_writes_separate_category_dataset_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            species_dir = input_root / "arabidopsis_thaliana"
            species_dir.mkdir(parents=True)
            output_root = root / "output"
            write_source(species_dir, "nuclear_intergenic", "intergenic_locus", 10)
            write_source(
                species_dir, "nuclear_pseudogenes", "annotated_pseudogene", 20
            )
            summary = convert_species(
                input_root,
                output_root,
                "arabidopsis_thaliana",
                (0.8, 0.1, 0.1),
                42,
                12,
            )
            self.assertEqual(summary["separate_datasets"]["intergenic"]["total"], 10)
            self.assertEqual(summary["separate_datasets"]["pseudogenes"]["total"], 20)
            for variant, expected_label in (("intergenic", 0), ("pseudogenes", 1)):
                for split in ("train", "validation", "test"):
                    frame = pd.read_parquet(
                        output_root / variant / "arabidopsis_thaliana" / f"{split}.parquet"
                    )
                    self.assertEqual(set(frame["binary_label"]), {expected_label})


if __name__ == "__main__":
    unittest.main()
