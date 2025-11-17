import argparse
import os
import re
from Bio import SeqIO


accepted_files = [
    '.fasta', 
    '.fa', 
    '.fna',
    '.fas'
    ]

def extract_sequences(file_type, file_path):
    """
    Extract sequences from a genomic data file.
    returns:
        {sequence_id: sequence_string}
    """
    sequences = {}
    if file_type == 'fasta':
        for record in SeqIO.parse(file_path, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences



def test_consecutive_n(sequence, k_consecutive_n, min_split_length):
    """ 
    Tests splitting on consecutive k ambiguous nucleotides "N" in a sequence.
        prints:
            num_splits: Number of sequences you'll have after splitting by consecutive k Ns.
            mean_length: Mean length of sequences after splitting by consecutive k Ns.
            median_length: Median length of sequences after splitting by consecutive k Ns.
            max_length: Maximum length of sequences after splitting by consecutive k Ns.
            min_length: Minimum length of sequences after splitting by consecutive k Ns.
            num_sequences_above_16384: Number of sequences longer than 16,384 after splitting.
            percentage_kept: Percentage of original sequence length kept after splitting.
    """
    seq_len = len(sequence)
    parts = re.split(f'N{{{k_consecutive_n},}}', sequence)
    split_sequences = [part for part in parts if len(part) >= min_split_length]
    split_lengths = [len(seq) for seq in split_sequences]
    num_splits = len(split_sequences)
    if num_splits == 0:
        print("No sequences found after splitting.")
        return

    mean_length = sum(split_lengths) / num_splits
    median_length = sorted(split_lengths)[num_splits // 2] if num_splits % 2 == 1 else \
        (sorted(split_lengths)[num_splits // 2 - 1] + sorted(split_lengths)[num_splits // 2]) / 2
    max_length = max(split_lengths)
    min_length = min(split_lengths)
    num_sequences_above_16384 = sum(1 for length in split_lengths if length > 16384)
    total_kept_length = sum(split_lengths)
    percentage_kept = (total_kept_length / seq_len) * 100  

    print(f"Number of splits: {num_splits}")
    print(f"Mean length: {mean_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"Max length: {max_length}")
    print(f"Min length: {min_length}")
    print(f"Number of sequences above 16,384: {num_sequences_above_16384}")
    print(f"Percentage of original sequence length kept: {percentage_kept:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run data processing tests on genomic sequences.")

    parser.add_argument("--filepath", type=str, default='../shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas', help="File path to the genomic data file.")
    parser.add_argument("--k_consecutive_n", type=int, default=3, help="Test splitting on consecutive k ambiguous nucleotides 'N'.")
    parser.add_argument("--min_split_length", type=int, default=16384, help="Test GC content calculation.")

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a valid file path using --filepath")
        return
    
    if not os.path.isfile(args.filepath):
        print(f"File not found: {args.filepath}")
        return
    
    _, ext = os.path.splitext(args.filepath)
    file_type = None
    if ext.lower() in accepted_files:
        if ext.lower() in ['.fasta', '.fa', '.fna', '.fas']:
            file_type = 'fasta'
    else:
        print(f"Unsupported file type: {ext}")
        return

    # sequences structure is {sequence_id: sequence_string}
    sequences = extract_sequences(file_type, args.filepath)

    for seq_id, sequence in sequences.items():
        print(f"\n--- Processing Sequence ID: {seq_id} | Length: {len(sequence)} ----------")
        test_consecutive_n(sequence, args.k_consecutive_n, args.min_split_length)

    return


if __name__ == "__main__":
    main()
