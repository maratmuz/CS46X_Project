from Bio.Align import substitution_matrices
import subprocess
import re
import os
from typing import List, Optional
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
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
            'gene_completion',
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
            tests = run.tests
            # repetitions = run.test.repetitions

            log_dir = out_root / f"run_{run_idx}" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # try:
            #     eval_tests = run.eval[f'{eval_mode}_tests']
            # except KeyError:
            #     raise ValueError(f"Could not find any defined tests in the config file, check the desired format and try again.")

            fasta_path = run.data.fasta_path
            gff_path = run.data.gff_path

            self._data_loader.load(
                fasta_path=fasta_path,
                gff_path=gff_path,
            )

            for model_idx, model_type in enumerate(model_types):
                with open(log_dir / "model_load.log", "a") as f, \
                    contextlib.redirect_stdout(f), \
                    contextlib.redirect_stderr(f):

                    model = self._load_model(model_type)

                results[model_type] = {}

                for test_idx, test in enumerate(tests.values()):
                    reps = test.reps
                    max_pred_len = test.max_pred_len

                    for chr_idx in range(self._data_loader.num_chromosomes()):
                        test_label = f'chromosome_{chr_idx + 1}'
                        results[model_type][test_label] = []

                        genes = self._data_loader.sample_genes(chr_id=chr_idx, count=reps)

                        for gene_data in genes:
                            input = gene_data['input']
                            label = gene_data['label']

                            try:
                                with torch.inference_mode():
                                    with open(log_dir / "model_output.log", "a") as f, \
                                        contextlib.redirect_stdout(f), \
                                        contextlib.redirect_stderr(f):
                                        output = model.generate(
                                            prompt_seqs=[str(input)[1500:]],
                                            n_tokens=len(label),
                                            temperature=0.7,
                                            top_k=4,
                                            # top_p=1.0,
                                        )

                            except Exception as e:
                                err_file = log_dir / "errors.log"
                                with open(err_file, "a") as ef:
                                    error = f"[ERROR] model.generate failed for {model_type}, test {test_label}: {e}\n"
                                    ef.write(error)
                                    print(error)
                                continue  # skip to next repetition

                            pred_dna = str(output.sequences[0])
                            label_dna = str(label)

                            pred_protein = self.extract_protein_with_augustus(str(input) + pred_dna, species="arabidopsis")
                            label_protein = self.extract_protein_with_augustus(str(input) + label_dna, species="arabidopsis")

                            res = self._analyze_alignments([pred_protein], [label_protein])
                            # self.get_best_protein_alignment(label, pred_seq)
                            pass
                            

                            # acc, matches = self._sequence_identity(label, pred_seq)
                            # results[model_type][test_label].append(acc)

                # Clear output tensors
                del output, pred_seq
                del model
                gc.collect()
                torch.cuda.empty_cache()

                        # if (rep_idx + 1) % 4 == 0:
                        #     print(
                        #         f'{"Model:":<11}1 / 1 | {model_type:<20}\n'
                        #         f'{"Test:":<11}1 / 1 | {test_label:<20}\n'
                        #         f'{"Repetition:":<11}{rep_idx + 1:>2} / {repetitions:<2}\n'
                        #     )


            # self._save_results(results, out_root, run_idx)

        pass

    def run_single(self, model_type, data_path, data_format, seq_len, pred_len, repetitions):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("output") / "eval" / f"results_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)

        results = {}

        log_dir = out_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._data_loader.load(
            path=data_path,
            format=data_format,
            verbose=True,
        )

        model = None
        try:
            with open(log_dir / "model_load.log", "a") as f, \
                contextlib.redirect_stdout(f), \
                contextlib.redirect_stderr(f):

                model = self._load_model(model_type)
        except Exception as e:
            err_file = log_dir / "errors.log"
            with open(err_file, "a") as ef:
                error = f"[ERROR] Failed to load model {model_type}: {e}\n"
                ef.write(error)
                print(error)
            return

        results[model_type] = {}

        test_label = f'SeqL {seq_len}, PredL {pred_len}'
        results[model_type][test_label] = []

        # Mode 2: Unique samples, start from position 0
        num_samples = min(repetitions, len(self._data_loader._data))
        self._data_loader.initialize_unique_samples(num_samples=num_samples)
        for rep_idx in range(repetitions):
            input, label = self._data_loader.read_start(
                splits=[seq_len, pred_len],
            )

            try:
                with torch.inference_mode():
                    with open(log_dir / "model_output.log", "a") as f, \
                        contextlib.redirect_stdout(f), \
                        contextlib.redirect_stderr(f):
                        output = model.generate(
                            prompt_seqs=[input],
                            n_tokens=pred_len,
                            temperature=1.0,
                            top_k=1,
                            top_p=1.0,
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

            # Clear output tensors
            del output, pred_seq

            if (rep_idx + 1) % 4 == 0:
                print(
                    f'{"Model:":<11}1 / 1 | {model_type:<20}\n'
                    f'{"Test:":<11}1 / 1 | {test_label:<20}\n'
                    f'{"Repetition:":<11}{rep_idx + 1:>2} / {repetitions:<2}\n'
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        self._save_results(results, out_root, 0)
        return out_root
    
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
        
    def _analyze_alignments(self, 
                        generated_seqs: List[str],
                        target_seqs: List[str],
                        names: Optional[List[str]] = None
                        ) -> List[dict]:
        """
        Analyze and visualize alignments between generated and target sequences.
        
        Args:
            generated_seqs: List of generated sequences
            target_seqs: List of target sequences
            names: Optional list of sequence names
            
        Returns:
            List of alignment metrics for each sequence pair
        """
        metrics = []
        print("\nSequence Alignments:")
        
        for i, (gen_seq, target_seq) in enumerate(zip(generated_seqs, target_seqs)):
            if names and i < len(names):
                print(f"\nAlignment {i+1} ({names[i]}):")
            else:
                print(f"\nAlignment {i+1}:")
            
            gen_bio_seq = Seq(gen_seq)
            target_bio_seq = Seq(target_seq)
            
            # Get alignments
            alignments = pairwise2.align.globalms(
                gen_bio_seq, target_bio_seq,
                match=2,
                mismatch=-1,
                open=-0.5,
                extend=-0.1
            )
            
            best_alignment = alignments[0]
            print(format_alignment(*best_alignment))
            
            matches = sum(a == b for a, b in zip(best_alignment[0], best_alignment[1]) 
                        if a != '-' and b != '-')
            alignment_length = len(best_alignment[0].replace('-', ''))
            similarity = (matches / len(target_seq)) * 100
            
            seq_metrics = {
                'similarity': similarity,
                'score': best_alignment[2],
                'length': len(target_seq),
                'gaps': best_alignment[0].count('-') + best_alignment[1].count('-')
            }
            
            if names and i < len(names):
                seq_metrics['name'] = names[i]
                
            metrics.append(seq_metrics)
            
            print(f"Sequence similarity: {similarity:.2f}%")
            print(f"Alignment score: {best_alignment[2]:.2f}")
        
        return metrics


    def get_best_protein_alignment(self, label_seq, pred_seq_str):
        # 1. Translate the Label (assume standard frame 0)
        label_protein = label_seq.translate(to_stop=False)
        
        # 2. Translate the Prediction in ALL 3 Forward Frames
        pred_frames = []
        pred_frames.append(Seq(pred_seq_str).translate(to_stop=False))      # Frame 0
        pred_frames.append(Seq(pred_seq_str[1:]).translate(to_stop=False))  # Frame 1
        pred_frames.append(Seq(pred_seq_str[2:]).translate(to_stop=False))  # Frame 2
        
        best_score = -1
        best_frame = -1
        
        print(f"Label Protein Len: {len(label_protein)}")
        
        # 3. Compare Label vs All 3 Prediction Frames
        for i, p_seq in enumerate(pred_frames):
            alignments = pairwise2.align.globalms(
                label_protein, p_seq,
                match=2, mismatch=-1, open=-0.5, extend=-0.1
            )
            score = alignments[0][2]
            print(f"Frame {i} Score: {score}")
            
            if score > best_score:
                best_score = score
                best_frame = i
                
        print(f"\nBest Frame was: {best_frame} with Score: {best_score}")

    def extract_protein_with_augustus(self, dna_seq: str, species="arabidopsis") -> str:
        """
        Runs Augustus on a raw DNA string to predict the gene and extract the 
        spliced/translated protein sequence.
        """
        # Define temp filenames
        input_file = "temp_augustus_input.fasta"
        
        # 1. Write the raw DNA to a FASTA file
        with open(input_file, "w") as f:
            f.write(f">seq1\n{dna_seq}\n")

        # 2. Run Augustus command
        # We capture stdout because Augustus prints the prediction there
        cmd = ["augustus", f"--species={species}", input_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[AUGUSTUS ERROR] Tool failed: {e}")
            return ""
        except FileNotFoundError:
            print("[AUGUSTUS ERROR] 'augustus' executable not found in PATH.")
            return ""
        finally:
            # Clean up temp file
            if os.path.exists(input_file):
                os.remove(input_file)

        # 3. Parse the output to find the protein sequence
        # Augustus provides the translation in a comment block: # protein sequence = [MKL...]
        # We use Regex to grab everything between the brackets
        match = re.search(r"# protein sequence = \[(.*?)\]", output, re.DOTALL)
        
        if match:
            raw_protein = match.group(1)
            # Clean up: Remove newlines, spaces, and the comment characters that might be inside
            clean_protein = raw_protein.replace("\n", "").replace(" ", "").replace("#", "")
            return clean_protein
        else:
            # Augustus failed to find a gene structure in this sequence
            return ""