
import sys
from Bio import SeqIO

def parse_gff_genes(gff_path):
    """
    Parses a GFF file and returns a list of (start, end, strand) tuples for all 'gene' features.
    Assumes 1-based inclusive coordinates (standard GFF).
    """
    genes = []
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            # Filter for genes on ChrM
            seqid = parts[0]
            feature_type = parts[2]
            
            if seqid != "ChrM":
                continue
                
            if feature_type == "gene" or feature_type == "Helixer gene": # Handle both standard and Helixer types
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[6]
                genes.append((start, end, strand))
    return genes

def calculate_overlap(pred_genes, truth_genes):
    """
    Calculates how many truth genes are 'covered' by predicted genes.
    Criterion: A truth gene is covered if any predicted gene overlaps it by at least 1bp.
    (This is a loose metric for simple benchmarking).
    """
    covered_count = 0
    
    # Sort for potential optimization, but O(N*M) is fine for small ChrM
    for t_start, t_end, t_strand in truth_genes:
        is_covered = False
        for p_start, p_end, p_strand in pred_genes:
            # Check overlap
            if max(t_start, p_start) <= min(t_end, p_end):
                is_covered = True
                break
        if is_covered:
            covered_count += 1
            
    return covered_count

def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark.py <prediction.gff> <ground_truth.gff>")
        sys.exit(1)
        
    pred_gff = sys.argv[1]
    truth_gff = sys.argv[2]
    
    print(f"Loading predictions from: {pred_gff}")
    preds = parse_gff_genes(pred_gff)
    print(f"Found {len(preds)} predicted genes on ChrM.")
    
    print(f"Loading ground truth from: {truth_gff}")
    truth = parse_gff_genes(truth_gff)
    print(f"Found {len(truth)} ground truth genes on ChrM.")
    
    if len(truth) == 0:
        print("No ground truth genes found. Cannot benchmark.")
        sys.exit(0)
        
    covered = calculate_overlap(preds, truth)
    sensitivity = (covered / len(truth)) * 100
    
    print("-" * 30)
    print(f"Benchmarking Results (ChrM):")
    print(f"Ground Truth Genes: {len(truth)}")
    print(f"Predicted Genes:    {len(preds)}")
    print(f"Recovered Genes:    {covered}")
    print(f"Sensitivity:        {sensitivity:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()
