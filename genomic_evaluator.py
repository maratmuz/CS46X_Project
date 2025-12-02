import contextlib
import gc
import torch
import pandas as pd 
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from genomic_data_loader import GenomicDataLoader
from Bio.Seq import Seq
from Bio import pairwise2

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
            'gene_pred',
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

            log_dir = out_root / f"run_{run_idx}" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            data_path = run.data.path
            data_format = run.data.format
            gff_path = run.data.get('gff_path', None)

            # Handle gene_pred mode separately (doesn't need repitions or eval_tests)
            if eval_mode == 'gene_pred':
                self._run_gene_pred_evaluation(
                    run, model_types, None, data_path, data_format,
                    out_root, run_idx, log_dir
                )
                continue
            
            # For seq_pred mode, get repitions and eval_tests
            repitions = run.eval.get('repitions', None)
            if repitions is None:
                raise ValueError(f"Missing required 'repitions' key for {eval_mode} mode in config.")
            
            try:
                eval_tests = run.eval[f'{eval_mode}_tests']
            except KeyError:
                raise ValueError(f"Could not find required '{eval_mode}_tests' in the config file, check the desired format and try again.")

            self._data_loader.load(
                path=data_path,
                format=data_format,
                gff_path=gff_path,
                verbose=True,
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

                    # Different read modes (comment/uncomment to switch):

                    # Mode 1: Random samples, random start positions
                    # for rep_idx in range(repitions):
                    #     input, label = self._data_loader.read(
                    #         splits=[seq_len, pred_len],
                    #     )

                    # Mode 2: Unique samples, start from position 0
                    num_samples = min(repitions, len(self._data_loader._data))
                    self._data_loader.initialize_unique_samples(num_samples=num_samples)
                    for rep_idx in range(repitions):
                        input, label = self._data_loader.read_start(
                            splits=[seq_len, pred_len],
                        )

                    # Mode 3: Midpoint split (matches test_evo2_generation.py)
                    # Note: seq_len from config is ignored; prompt length is 50% of actual sequence
                    # num_samples = min(repitions, len(self._data_loader._data))
                    # self._data_loader.initialize_unique_samples(num_samples=num_samples)

                    # for rep_idx in range(repitions):
                    #     input, label = self._data_loader.read_midpoint(pred_len=pred_len)

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
                    f'{"Repetition:":<11}{rep_idx + 1:>2} / {repitions:<2}\n'
                )

                del model
                gc.collect()
                torch.cuda.empty_cache()

            self._save_results(results, out_root, run_idx)

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

        # Separate chromosome-level and gene-level results
        gene_results = {}
        chromosome_results = {}
        
        for model, tests in results.items():
            gene_results[model] = {}
            chromosome_results[model] = {}
            for test_label, vals in tests.items():
                if test_label.startswith('Chromosome_'):
                    chromosome_results[model][test_label] = vals
                else:
                    gene_results[model][test_label] = vals

        # Average results table (gene-level) - includes recovery, expected_min, and model_contribution
        if gene_results:
            df_avg = pd.DataFrame({
                model: {
                    test: (sum(vals) / len(vals) if vals else float('nan'))
                    for test, vals in tests.items()
                }
                for model, tests in gene_results.items()
            })
            df_avg.to_csv(base / "avg_results.csv", index=True)

        # Chromosome-level average results table
        if chromosome_results:
            df_chrom_avg = pd.DataFrame({
                model: {
                    test: (sum(vals) / len(vals) if vals else float('nan'))
                    for test, vals in tests.items()
                }
                for model, tests in chromosome_results.items()
            })
            df_chrom_avg.to_csv(base / "avg_results_by_chromosome.csv", index=True)
            
            # Print chromosome averages to console
            print("\n" + "=" * 80)
            print("AVERAGE PROTEIN RECOVERY BY CHROMOSOME")
            print("=" * 80)
            for model in df_chrom_avg.columns:
                print(f"\n{model}:")
                for chrom in df_chrom_avg.index:
                    avg = df_chrom_avg.loc[chrom, model]
                    if pd.notna(avg):
                        chrom_name = chrom.replace('Chromosome_', '')
                        print(f"  {chrom_name}: {avg:.2f}%")
            print("=" * 80 + "\n")

        # per-test tables (gene-level)
        if not gene_results:
            return
        all_tests = sorted({t for tests in gene_results.values() for t in tests})

        for i, test_label in enumerate(all_tests, start=1):
            df_full = pd.DataFrame({
                model: pd.Series(tests.get(test_label, []))
                for model, tests in gene_results.items()
            })
            df_full.index.name = "rep"
            df_full.to_csv(full_dir / f"test_{i}.csv", index=True)

    def _run_gene_pred_evaluation(self, run, model_types, eval_tests, data_path, 
                                   data_format, out_root, run_idx, log_dir):
        """
        Run gene prediction evaluation matching the paper methodology.
        
        For each gene:
        - Build prompt: 1kb upstream + first 500/1000bp of CDS
        - Generate 10 samples with temperature=0.7, top_k=4
        - Translate to amino acids and align to natural protein
        - Calculate percent protein recovery
        """
        results = {}
        
        # Get configuration parameters
        organism_type = run.data.get('organism_type', 'prokaryote')
        num_genes = run.eval.get('num_genes', None)
        samples_per_prompt = run.eval.get('samples_per_prompt', 10)
        chromosomes = run.data.get('chromosomes', None)  # List of chromosomes to evaluate
        
        # Load data
        gff_path = run.data.get('gff_path', None)
        self._data_loader.load(
            path=data_path, 
            format=data_format, 
            gff_path=gff_path,
            verbose=True
        )
        
        # Get total number of genes available
        total_genes_available = len(self._data_loader._genes) if self._data_loader._genes else 0
        
        # Initialize gene evaluation with chromosome filtering
        print(f"\n{'='*80}")
        print("INITIALIZING GENE EVALUATION")
        print(f"{'='*80}")
        print(f"Organism type: {organism_type}")
        print(f"Chromosomes: {chromosomes if chromosomes else 'All'}")
        print(f"Total genes available (across all chromosomes): {total_genes_available:,}")
        print(f"Number of genes to evaluate: {num_genes if num_genes else 'All available (after filtering)'}")
        print(f"Samples per prompt: {samples_per_prompt}")
        print(f"Models to evaluate: {len(model_types)} ({', '.join(model_types)})")
        print(f"{'='*80}\n")
        
        # Determine minimum CDS length based on organism type (seed length requirement)
        min_cds_length = 500 if organism_type.lower() in ['prokaryote', 'archaea', 'yeast'] else 1000
        
        self._data_loader.initialize_gene_evaluation(
            num_genes=num_genes,
            chromosomes=chromosomes,
            min_cds_length=min_cds_length
        )
        
        num_genes_to_process = len(self._data_loader._selected_genes)
        print(f"Selected {num_genes_to_process:,} genes for evaluation (out of {total_genes_available:,} available)\n")
        
        for model_idx, model_type in enumerate(model_types):
            print(f"\n{'='*80}")
            print(f"Model: {model_idx + 1}/{len(model_types)} | {model_type}")
            print(f"{'='*80}\n")
            
            with open(log_dir / "model_load.log", "a") as log_f, \
                contextlib.redirect_stdout(log_f), \
                contextlib.redirect_stderr(log_f):
                print(f"[{model_type}] Loading model...")
                model = self._load_model(model_type)
            
            print(f"[{model_type}] Model loaded successfully")
            results[model_type] = {}
            
            # Process each gene
            print(f"[{model_type}] Processing {num_genes_to_process} genes...\n")
            
            for gene_idx in range(num_genes_to_process):
                try:
                    # Get current gene info for error reporting (before calling get_gene_prompt)
                    current_gene = self._data_loader._selected_genes[gene_idx] if gene_idx < len(self._data_loader._selected_genes) else None
                    gene_id_for_error = current_gene.gene_id if current_gene else f"index_{gene_idx}"
                    
                    # Get gene prompt and target
                    prompt, target_cds_seq, full_cds, gene_ann = self._data_loader.get_gene_prompt(
                        organism_type=organism_type
                    )
                    
                    print(f"[{model_type}] Gene {gene_idx + 1}/{num_genes_to_process}: {gene_ann.gene_id} ({gene_ann.seq_id})")
                    print(f"  Prompt length: {len(prompt):,} bp, Target CDS length: {len(target_cds_seq):,} bp")
                    
                    # Generate samples
                    generated_samples = []
                    for sample_idx in range(samples_per_prompt):
                        try:
                            with torch.inference_mode():
                                output = model.generate(
                                    prompt_seqs=[prompt],
                                    n_tokens=len(target_cds_seq),  # Generate remainder of CDS sequence
                                    temperature=0.7,
                                    top_k=4,
                                    top_p=1.0,
                                )
                                generated_samples.append(output.sequences[0])
                        except Exception as e:
                            err_file = log_dir / "errors.log"
                            with open(err_file, "a") as ef:
                                error = f"[ERROR] Generation failed for {model_type}, gene {gene_ann.gene_id}, sample {sample_idx}: {e}\n"
                                ef.write(error)
                            continue
                    
                    if not generated_samples:
                        continue
                    
                    # Use frame 0 for translation
                    # Note: The CDS sequence extraction properly handles GFF3 phase information
                    # by trimming phase bases from each CDS feature to preserve codon boundaries
                    # across exon junctions. The resulting full_cds_seq is already in the correct
                    # reading frame (starts at codon boundary), so we translate from frame 0.
                    cds_phase = 0  # Properly assembled CDS sequences start at frame 0
                    
                    # Calculate expected minimum recovery from seed portion
                    # The seed (500-1000 bp) is in the prompt and will always match perfectly,
                    # so minimum recovery = (seed_protein_length / full_protein_length) * 100
                    # We translate both with to_stop=True to match the actual recovery calculation
                    cds_seed_len = 500 if organism_type.lower() in ['prokaryote', 'archaea', 'yeast'] else 1000
                    cds_seed_dna = full_cds[:cds_seed_len]  # First part of CDS (used as seed)
                    try:
                        seed_protein = Seq(cds_seed_dna).translate(to_stop=True)
                        full_protein = Seq(full_cds).translate(to_stop=True)
                        if len(full_protein) > 0:
                            expected_min_recovery = (len(seed_protein) / len(full_protein)) * 100
                        else:
                            expected_min_recovery = 0.0
                    except Exception as e:
                        expected_min_recovery = None
                        print(f"  Warning: Could not calculate expected minimum recovery: {e}")
                    
                    # Calculate protein recovery for each sample
                    protein_recoveries = []
                    for gen_idx, gen_seq in enumerate(generated_samples):
                        # Combine prompt CDS seed with generated sequence to get complete CDS
                        cds_seed = prompt[-cds_seed_len:]  # CDS seed from prompt
                        completed_cds = cds_seed + gen_seq[:len(target_cds_seq)]
                        
                        # Translate and calculate protein recovery using correct reading frame
                        # Extracted CDS sequences are in frame 0, so phase=0 is correct
                        print(f"Translating generated sample {gen_idx + 1}/{len(generated_samples)}, gene index {gene_idx}, gene {gene_ann.gene_id} ({gene_ann.seq_id})")
                        recovery = self._calculate_protein_recovery(completed_cds, full_cds, phase=cds_phase)
                        if recovery is not None:
                            protein_recoveries.append(recovery)
                    
                    # Store results (average recovery across samples for this gene)
                    if protein_recoveries:
                        gene_label = f"Gene_{gene_ann.gene_id}"
                        chromosome_id = gene_ann.seq_id
                        
                        # Gene-level results
                        if gene_label not in results[model_type]:
                            results[model_type][gene_label] = []
                        avg_recovery = sum(protein_recoveries) / len(protein_recoveries)
                        results[model_type][gene_label].append(avg_recovery)
                        
                        # Store expected minimum recovery and model contribution as separate metrics
                        if expected_min_recovery is not None:
                            expected_min_label = f"Gene_{gene_ann.gene_id}_expected_min"
                            if expected_min_label not in results[model_type]:
                                results[model_type][expected_min_label] = []
                            results[model_type][expected_min_label].append(expected_min_recovery)
                            
                            model_contribution = avg_recovery - expected_min_recovery
                            contribution_label = f"Gene_{gene_ann.gene_id}_model_contribution"
                            if contribution_label not in results[model_type]:
                                results[model_type][contribution_label] = []
                            results[model_type][contribution_label].append(model_contribution)
                            
                            print(f"  Average protein recovery: {avg_recovery:.2f}% (from {len(protein_recoveries)} samples)")
                            print(f"  Expected minimum (from seed): {expected_min_recovery:.2f}%")
                            print(f"  Model contribution: {model_contribution:+.2f}% (recovery above seed baseline)")
                        else:
                            print(f"  Average protein recovery: {avg_recovery:.2f}% (from {len(protein_recoveries)} samples)")
                        
                        # Chromosome-level results
                        chrom_label = f"Chromosome_{chromosome_id}"
                        if chrom_label not in results[model_type]:
                            results[model_type][chrom_label] = []
                        results[model_type][chrom_label].append(avg_recovery)
                        
                        # Save results after each gene to prevent data loss if cancelled
                        self._save_results(results, out_root, run_idx)
                    else:
                        print(f"  No valid protein recoveries calculated (translation/alignment failed)")
                    
                    # Progress summary every 10 genes
                    if (gene_idx + 1) % 10 == 0:
                        print(f"\n[{model_type}] Progress: {gene_idx + 1}/{num_genes_to_process} genes completed\n")
                
                except Exception as e:
                    err_file = log_dir / "errors.log"
                    with open(err_file, "a") as ef:
                        # Try to get gene_id from the exception or use current gene
                        gene_id = gene_id_for_error if 'gene_id_for_error' in locals() else f"index_{gene_idx}"
                        error = f"[ERROR] Gene processing failed for {model_type}, gene_idx {gene_idx} (gene_id: {gene_id}): {e}\n"
                        ef.write(error)
                        print(error)  # Also print to console for visibility
                    continue
            
            # Cleanup
            print(f"\n[{model_type}] Completed. Cleaning up model...")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{model_type}] Model cleanup complete\n")
        
        # Final save of all results
        print("Saving final results...")
        self._save_results(results, out_root, run_idx)
        print("Results saved successfully!")

    def _calculate_protein_recovery(self, generated_cds, natural_cds, phase=None):
        """
        Calculate percent protein recovery by:
        1. Translating DNA sequences to amino acids using the correct reading frame (phase)
        2. Aligning generated protein to natural protein
        3. Calculating sequence identity over aligned region
        
        Args:
            generated_cds: Generated CDS DNA sequence
            natural_cds: Natural CDS DNA sequence (already correctly phased from GFF)
            phase: Reading frame phase (0, 1, or 2) from GFF file. If None, tries all frames.
        
        Returns:
            Percent protein recovery (0-100) or None if translation fails
        """
        try:
            generated_seq = Seq(generated_cds)
            natural_seq = Seq(natural_cds)
            
            # Use phase from GFF if available, otherwise try all frames
            if phase is not None:
                frames_to_try = [int(phase)]
            else:
                frames_to_try = [0, 1, 2]
            
            generated_proteins = []
            natural_proteins = []
            
            for frame in frames_to_try:
                try:
                    # Translate from specified frame
                    gen_prot = generated_seq[frame:].translate(to_stop=True)
                    if len(gen_prot) > 0:
                        generated_proteins.append((frame, str(gen_prot)))
                except:
                    pass
                
                try:
                    nat_prot = natural_seq[frame:].translate(to_stop=True)
                    if len(nat_prot) > 0:
                        natural_proteins.append((frame, str(nat_prot)))
                except:
                    pass
            
            if not generated_proteins or not natural_proteins:
                return None
            
            # Use the longest translations (or the one matching the phase)
            generated_protein = max(generated_proteins, key=lambda x: len(x[1]))[1]
            natural_protein = max(natural_proteins, key=lambda x: len(x[1]))[1]
            
            # Perform global alignment with specific scoring parameters
            alignments = pairwise2.align.globalms(
                generated_protein,
                natural_protein,
                match=2,
                mismatch=-1,
                open=-0.5,
                extend=-0.1
            )
            
            if not alignments:
                return None
            
            best_alignment = alignments[0]
            aligned_gen = best_alignment[0]  # seqA is first element of tuple
            aligned_nat = best_alignment[1]  # seqB is second element of tuple

            print("\n===== BEST ALIGNMENT =====")
            print(pairwise2.format_alignment(*best_alignment))
            print("==========================\n")
            
            # Calculate matches: only count positions where both sequences have non-gap characters
            matches = sum(a == b for a, b in zip(aligned_gen, aligned_nat) 
                         if a != '-' and b != '-')
            
            # Similarity is calculated as matches / target sequence length (not alignment length)
            target_length = len(natural_protein)
            
            if target_length == 0:
                return 0.0
            
            recovery = (matches / target_length) * 100
            return recovery
            
        except Exception as e:
            # Return None if translation/alignment fails
            return None