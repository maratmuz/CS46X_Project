import contextlib
import gc
import torch
import pandas as pd 
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from genomic_data_loader import GenomicDataLoader

from evo2 import Evo2


class GenomicEvaluator:
    def __init__(self):
        self.supported_models = {
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

        self.supported_eval_modes = [
            'seq_pred',
        ]

        # self._model_manager = GenomicModelManager()
        self._data_loader = GenomicDataLoader()

    def run(self, config_path):
        config = OmegaConf.load(config_path)
        runs = config.runs

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("output") / "eval" / f"results_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)


        for run_idx, run in enumerate(runs.values()):
            results = {}

            model_types = run.model_types
            eval_mode = run.eval.mode
            repitions = run.eval.repitions

            log_dir = out_root / f"run_{run_idx}" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            try:
                eval_tests = run.eval[f'{eval_mode}_tests']
            except KeyError:
                raise ValueError(f"Could not find any defined tests in the config file, check the desired format and try again.")

            data_path = run.data.path
            data_format = run.data.format

            self._data_loader.load(
                path=data_path, 
                format=data_format,
            )

            for model_idx, model_type in enumerate(model_types):
                with open(log_dir / "model_load.log", "a") as f, \
                    contextlib.redirect_stdout(f), \
                    contextlib.redirect_stderr(f):

                    model = self._load_model(model_type)

                results[model_type] = {}

                for test_idx, test in enumerate(eval_tests.values()):
                    seq_len = test.seq_len
                    pred_len = test.pred_len

                    test_label = f'SeqL {seq_len}, PredL {pred_len}'
                    results[model_type][test_label] = []

                    # Random samples, random start positions
                    # for rep_idx in range(repitions): 
                    #     input, label = self._data_loader.read(
                    #         splits=[
                    #             seq_len, 
                    #             pred_len,
                    #         ],
                    #     )
                    
                    # Initialize unique samples for this test (up to repitions count)
                    # If repitions > dataset size, will cycle through available samples
                    num_samples = min(repitions, len(self._data_loader._data))
                    self._data_loader.initialize_unique_samples(num_samples=num_samples)

                    for rep_idx in range(repitions): 
                        input, label = self._data_loader.read_start(
                            splits=[
                                seq_len, 
                                pred_len,
                            ],
                        )

                        try:
                            with open(log_dir / "model_output.log", "a") as f, \
                                contextlib.redirect_stdout(f), \
                                contextlib.redirect_stderr(f):
                                output = model.generate(
                                    prompt_seqs=[input],
                                    n_tokens=pred_len,
                                    temperature=1.0,
                                    top_k=4
                                )

                        except Exception as e:
                            err_file = log_dir / "errors.log"
                            with open(err_file, "a") as ef:
                                error = f"[ERROR] model.generate failed for {model_type}, test {test_label}, rep {rep_idx}: {e}\n"
                                ef.write(error)
                                print(error)
                            continue  # skip to next repetition

                        pred_seq = output.sequences[0]
                        acc, matches = self._sequence_identity(label, pred_seq)
                        results[model_type][test_label].append(acc) 

                        if (rep_idx + 1) % 4 == 0:
                            print(
                                f'{"Model:":<11}{model_idx + 1:>2} / {len(model_types):<2} | {model_type:<20}\n'
                                f'{"Test:":<11}{test_idx + 1:>2} / {len(eval_tests.values()):<2} | {test_label:<20}\n'
                                f'{"Repetition:":<11}{rep_idx + 1:>2} / {repitions:<2}\n'
                            )

                del model
                gc.collect()
                torch.cuda.empty_cache()

            self._save_results(results, out_root, run_idx)

        pass
    
    def _sequence_identity(self, seq_a: str, seq_b: str) -> tuple[float, int]:
        assert len(seq_a) == len(seq_b), "Sequences must be of equal length"

        matches = 0
        for base_a, base_b in zip(seq_a, seq_b):
            if base_a == base_b:
                matches += 1

        accuracy = matches / len(seq_a)
        return accuracy, matches

    def _load_model(self, model_type):
        # Checks every list in self.supported_models
        if not any(model_type in models for models in self.supported_models.values()):
            raise ValueError(
                f'Model type {model_type} was not found in the list of supported models.'
            )

        elif model_type in self.supported_models['Evo2']:
            model = Evo2(model_type)

        elif model_type in self.supported_models['PlantCAD2']:
            pass

        return model
    
    def _save_results(self, results: dict, out_root: Path, run_idx: int):
        base = out_root / f"run_{run_idx}"
        full_dir = base / "full_results"
        full_dir.mkdir(parents=True, exist_ok=True)

        # average results table
        df_avg = pd.DataFrame({
            model: {
                test: (sum(vals) / len(vals) if vals else float('nan'))
                for test, vals in tests.items()
            }
            for model, tests in results.items()
        })
        df_avg.to_csv(base / "avg_results.csv", index=True)  # CSV now

        # per-test tables
        if not results:
            return
        all_tests = sorted({t for tests in results.values() for t in tests})

        for i, test_label in enumerate(all_tests, start=1):
            df_full = pd.DataFrame({
                model: pd.Series(tests.get(test_label, []))
                for model, tests in results.items()
            })
            df_full.index.name = "rep"
            df_full.to_csv(full_dir / f"test_{i}.csv", index=True)