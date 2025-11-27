import os
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq

from evo2 import Evo2

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)



model_name = 'evo2_7b'

model = Evo2(model_name)




def read_sequences(input_file: Path) -> Tuple[List[str], List[str]]:
    """
    Read input and target sequences from CSV file.
    
    Expected CSV format:
    input_sequence,target_sequence
    ACGTACGT,ACGTACGTAA
    ...
    """
    input_seqs: List[str] = []
    names: List[str] = []
    
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            input_seqs.append(row[0])
            if len(row) > 1:
                names.append(row[1])
    
    return input_seqs, names

# Load example data

sequences, names = read_sequences('models/Evo2/evo2/test/data/prompts.csv')

# For 'autocomplete', we split the data into input and target sequences

input_seqs = [seq[:500] for seq in sequences]
target_seqs = [seq[500:1000] for seq in sequences]

print(f"Loaded {len(sequences)} sequence pairs")


generations = model.generate(
    input_seqs,
    n_tokens=500,
    temperature=1.0,
)

generated_seqs = generations.sequences
print(generated_seqs)

def analyze_alignments(generated_seqs: List[str],
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

# Analyze alignments
alignment_metrics = analyze_alignments(generated_seqs, target_seqs, names)


pass