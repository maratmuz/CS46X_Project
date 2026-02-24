import argparse
import os
import sys
import subprocess
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Run Helixer gene prediction on sequences.")
    parser.add_argument("--gff", help="Optional: Path to input GFF file defining regions to extract (sequence GFF).")
    parser.add_argument("--fasta", required=True, help="Path to reference FASTA file.")
    parser.add_argument("--output", required=True, help="Path to output GFF file.")
    parser.add_argument("--model", default="land_plant", help="Helixer model to use (default: land_plant).")
    return parser.parse_args()

def sanitize_sequence(seq_str):
    """
    Sanitizes a DNA sequence string for Helixer.
    - Removes whitespace.
    - Converts U -> T (RNA handling).
    - Replaces any character not in {A, C, G, T, N} with 'N'.
    """
    # 1. Basic cleanup and upper case
    clean = seq_str.upper().replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    return clean.replace("U", "T")
    # Note: Helixer seems strict. If we encounter ambiguous bases other than N, 
    # we might need to mask them. However, usually N is sufficient. 
    # For now, let's trust that U->T covers the main issue, but if we see 'P' etc,
    # we should likely map them to N.
    
    # More aggressive approach if needed:
    # allowed = set("ACGTN")
    # return "".join([c if c in allowed else "N" for c in clean.replace("U", "T")])

def extract_sequences(gff_path, fasta_path):
    """
    Extracts sequences from reference FASTA based on GFF regions.
    Returns a list of SeqRecord objects.
    """
    print(f"Loading reference FASTA: {fasta_path}")
    # Use index for efficient access to large genomes
    fasta_index = SeqIO.index(fasta_path, "fasta")
    
    extracted_records = []
    
    print(f"Parsing GFF: {gff_path}")
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
                
            seq_id = parts[0]
            start = int(parts[3]) - 1 # GFF is 1-based, Python is 0-based
            end = int(parts[4])     # GFF end is inclusive
            
            if seq_id not in fasta_index:
                print(f"Warning: Sequence ID {seq_id} not found in FASTA. Skipping.")
                continue
                
            # Extract sequence
            raw_seq = fasta_index[seq_id].seq[start:end]
            
            # Sanitize
            sanitized_seq_str = sanitize_sequence(str(raw_seq))
            
            # Create a record ID that encodes the execution context
            # Use a clean ID for Helixer
            record_id = f"{seq_id}_{start+1}_{end}"
            record = SeqRecord(Seq(sanitized_seq_str), id=record_id, description=f"Extracted from {seq_id}")
            extracted_records.append(record)
            
    return extracted_records

def sanitize_and_write_temp_fasta(input_fasta, output_temp_fasta):
    """
    Reads an input FASTA, sanitizes sequences (removes spaces/newlines, converts U->T, handles garbage),
    and writes to a temporary file. Useful for direct FASTA input mode.
    """
    print(f"Sanitizing input FASTA: {input_fasta}")
    sanitized_records = []
    # Using SeqIO.parse instead of index since we want to stream/process all
    for record in SeqIO.parse(input_fasta, "fasta"):
        # Aggressive sanitization
        # Based on errors: ' ' and 'P' seen.
        # We will use a filter to keep only ACGTN
        clean_basic = str(record.seq).upper().replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "").replace("U", "T")
        
        # Generator expression for efficiency
        allowed = set("ACGTN")
        clean_seq = "".join(c if c in allowed else "N" for c in clean_basic)
        
        sanitized_records.append(SeqRecord(Seq(clean_seq), id=record.id, description=record.description))
    
    if not sanitized_records:
         print("Warning: No sequences found during sanitization.")
         
    SeqIO.write(sanitized_records, output_temp_fasta, "fasta")
    print(f"Sanitized FASTA written to {output_temp_fasta}")

def main():
    args = parse_args()
    
    # Logic:
    # If GFF is provided: Extract sequences -> Temp FASTA -> Helixer
    # If NO GFF: Sanitize Input FASTA -> Temp FASTA -> Helixer
    
    temp_fasta = f"temp_extracted_{os.getpid()}.fasta"
    
    if args.gff:
        print(f"GFF provided. Extracting sequences from {args.fasta} based on {args.gff}...")
        records = extract_sequences(args.gff, args.fasta)
        
        if not records:
            print("No sequences extracted from GFF regions. Exiting.")
            sys.exit(0)
        
        print(f"Extracted {len(records)} sequences.")
        SeqIO.write(records, temp_fasta, "fasta")
        print(f"Temporary FASTA written to {temp_fasta}")
    else:
        print(f"No GFF provided. processing {args.fasta}...")
        # We MUST sanitize for HelixerLite to avoid KeyError on spaces
        sanitize_and_write_temp_fasta(args.fasta, temp_fasta)

    # 3. Run Helixer CLI
    print(f"Running Helixer with model: {args.model}")
    
    cmd = [
        sys.executable, "-m", "helixerlite",
        "--fasta", temp_fasta,
        "--out", args.output,
        "--lineage", args.model
    ]
    
    # Pass environment variables to limit threads and prevent crashes
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, env=env)
        print("Helixer prediction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing Helixer: {e}")
        # Cleanup
        if temp_fasta and os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if temp_fasta and os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        sys.exit(1)

    # Cleanup
    if temp_fasta and os.path.exists(temp_fasta):
        os.remove(temp_fasta)
        
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
