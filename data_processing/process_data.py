import os
import re
import yaml
import argparse
from pathlib import Path
from Bio import SeqIO

accepted_files = {
    "fasta": ['.fasta', '.fa', '.fna', '.fas'],
    # "gff": ['.gff', '.gff3'],
    # "fastq": ['.fastq', '.fq'],
}


def split_sequence_on_consecutive_n(sequence, config):
    """
    Splits a sequence on consecutive k ambiguous nucleotides "N".
    returns:
        List of sequences after splitting.
    """
    k_consecutive_n = config.get('k_consecutive_n', 3)
    min_split_length = config.get('min_split_length', 16384)
    parts = re.split(f'N{{{k_consecutive_n},}}', sequence)
    split_sequences = [part for part in parts if len(part) >= min_split_length]
    return split_sequences


def process_fasta(filepath, output_path, data_config):
    # sequences structure is {sequence_id: sequence_string}
    sequences = {}
    for record in SeqIO.parse(filepath, "fasta"):
        sequences[record.id] = str(record.seq)

    os.makedirs(output_path.parent, exist_ok=True)
    if output_path.exists():
        with open(output_path, 'w') as out_f:
                out_f.write("")  # Clear existing file

    for seq_id, sequence in sequences.items():
        sequence_len = len(sequence)
        split_seqs = split_sequence_on_consecutive_n(sequence, data_config)
        kept_length = sum(len(seq) for seq in split_seqs)
        percentage_kept = round((kept_length / sequence_len) * 100, 2)
        # Save or further process split_seqs as needed
        if len(split_seqs) == 0:
            print(f"No sequences found after splitting for {seq_id}.")
            continue
        if len(split_seqs) > 1:
            for i, split_seq in enumerate(split_seqs):
                new_seq_id = f"{seq_id}.{i+1}"
                split_len = len(split_seq)
                percentage_of_og = round((split_len / sequence_len) * 100, 2)
                with open(output_path, 'a') as out_f:
                    out_f.write(f">{new_seq_id} Length: {split_len} {percentage_of_og}% of {seq_id} (Kept: {percentage_kept}%)\n{split_seq}\n")
        else:
            with open(output_path, 'a') as out_f:
                out_f.write(f">{seq_id}\n{split_seqs[0]}\n")

    print(f"Processed FASTA file. Output saved to {output_path}")
    return

        

def build_output_path(config_name, input_path):
    input_path = Path(input_path)

    # Convert to parts for manipulation
    parts = list(input_path.parts)

    # Find the index of "datasets"
    try:
        idx = parts.index("datasets")
    except ValueError:
        raise ValueError("Input path must contain a 'datasets' directory.")

    # Build new parts:
    # everything up to "datasets"  +
    # ["datasets", "processed", config_name] +
    # everything after "datasets"
    new_parts = (
        parts[: idx + 1] +
        ["processed", config_name] +
        parts[idx + 1 :]
    )

    # Reconstruct as a Path
    return Path(*new_parts)


def process(filepath, data_config):
    with open("data_configs.yml", 'r') as file:
        config_file = yaml.safe_load(file)
    config = config_file.get(data_config, {})

    if filepath is None:
        print("Please provide a valid file path using --filepath")
        return
    
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    output_path = build_output_path(data_config, filepath)

    print(f"Processing file: {filepath}")
    print(f"Output will be saved to: {output_path}")

    _, ext = os.path.splitext(filepath)
    file_type = None
    if any(ext.lower() in extensions for extensions in accepted_files.values()):
        if ext.lower() in accepted_files['fasta']:
            print("Processing FASTA file.")
            process_fasta(filepath, output_path, config)
    else:
        print(f"Unsupported file type: {ext}")
        return

def main():
    parser = argparse.ArgumentParser(description="Process genomic data files based on configuration.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the genomic data file.")
    parser.add_argument("--data_config", type=str, required=True, help="Data processing configuration name.")
    args = parser.parse_args()
    process(args.filepath, args.data_config)

if __name__ == "__main__":
    main()