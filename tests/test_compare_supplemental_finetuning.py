import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_supplemental_finetuning import compare_pgb, compare_supplemental


class FineTuningComparisonTests(unittest.TestCase):
    def test_held_out_deltas_are_computed_from_matching_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            fine = root / "fine"
            original = source / "runs" / "evo2_7b"
            continued = fine / "runs" / "evo2_7b"
            old_metrics = original / "evaluation" / "arabidopsis_thaliana" / "test_metrics.json"
            new_metrics = continued / "evaluation" / "arabidopsis_thaliana" / "test_metrics.json"
            old_metrics.parent.mkdir(parents=True)
            new_metrics.parent.mkdir(parents=True)
            (continued / "run_config.json").write_text(
                json.dumps({
                    "model": "evo2", "size": "7b", "species": ["arabidopsis_thaliana"],
                    "source_run_dir": str(original),
                }), encoding="utf-8",
            )
            original_values = {
                "accuracy": 0.5, "balanced_accuracy": 0.4, "macro_f1": 0.3,
                "weighted_f1": 0.5, "mcc": 0.2,
            }
            improved_values = dict(original_values, macro_f1=0.35)
            old_metrics.write_text(json.dumps({"overall": original_values}), encoding="utf-8")
            new_metrics.write_text(json.dumps({"overall": improved_values}), encoding="utf-8")
            output = fine / "comparison"
            compare_pgb(source, fine, output)
            with (output / "baseline_vs_finetuned_pgb_test.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["head"], "evo2/7b")
            self.assertAlmostEqual(float(row["delta_macro_f1"]), 0.05)

    def test_supplemental_comparison_keeps_locus_categories_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fine = Path(temporary)
            for subdir, low in (("baseline_supplemental_test", 0.4), ("supplemental_test", 0.7)):
                destination = fine / subdir / "metrics_by_model_category.csv"
                destination.parent.mkdir()
                with destination.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "head", "locus_set", "low_call_rate", "medium_call_rate",
                        "high_call_rate", "mean_probability_low",
                    ])
                    writer.writeheader()
                    writer.writerow({
                        "head": "deepcre/cnn", "locus_set": "intergenic",
                        "low_call_rate": low, "medium_call_rate": 1 - low,
                        "high_call_rate": 0, "mean_probability_low": low,
                    })
            output = fine / "comparison"
            compare_supplemental(fine, output)
            with (output / "baseline_vs_finetuned_supplemental_test.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["locus_set"], "intergenic")
            self.assertAlmostEqual(float(row["delta_low_call_rate"]), 0.3)


if __name__ == "__main__":
    unittest.main()
