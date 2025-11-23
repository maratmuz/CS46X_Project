# Marat/gff_fasta_to_csv.py

"""
Create a CSV of annotated coding sequences from a genome FASTA and GFF3.

Usage from CS46X_Project root:

  python Marat/gff_fasta_to_csv.py

Paths are hard coded below; edit FASTA_PATH, GFF3_PATH, and OUTPUT_PATH if needed.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

# ==========================
# HARD-CODED PATHS / CONSTANTS
# ==========================

FASTA_PATH  = Path("shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas")
GFF3_PATH   = Path("shared/datasets/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff")
OUTPUT_PATH = Path("Marat/output/tair10_cds_annotations.csv")

class GFFFeature:
    """Simple container for one GFF3 row."""

    __slots__ = (
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    )

    def __init__(self, seqid, source, type_, start, end, score, strand, phase, attributes):
        self.seqid = seqid
        self.source = source
        self.type = type_
        self.start = int(start)
        self.end = int(end)
        self.score = score
        self.strand = strand
        self.phase = phase
        self.attributes = attributes

    def __repr__(self):
        return (
            f"GFFFeature(type={self.type}, seqid={self.seqid}, "
            f"start={self.start}, end={self.end}, strand={self.strand})"
        )


def parse_attributes(attr_str: str) -> dict:
    """
    Parse the 9th column of a GFF3 line into a dict.

    Example:
      'ID=cds1;Parent=tx1;Name=Example' -> {"ID": "cds1", "Parent": "tx1", "Name": "Example"}
    """
    attrs = {}
    if not attr_str or attr_str == ".":
        return attrs

    parts = attr_str.strip().split(";")
    for part in parts:
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            attrs[key] = value
        else:
            attrs[part] = True
    return attrs


def read_gff3(path: Path) -> list["GFFFeature"]:
    """Parse a GFF3 file into a list of GFFFeature objects."""
    features: list[GFFFeature] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) != 9:
                continue  # skip malformed lines

            seqid, source, type_, start, end, score, strand, phase, attr_str = cols
            attrs = parse_attributes(attr_str)
            features.append(
                GFFFeature(
                    seqid=seqid,
                    source=source,
                    type_=type_,
                    start=start,
                    end=end,
                    score=score,
                    strand=strand,
                    phase=phase,
                    attributes=attrs,
                )
            )
    return features


def build_gene_and_transcript_maps(features: list["GFFFeature"]):
    """
    Build:
      - genes: gene_id -> info dict
      - transcripts: transcript_id -> info dict
      - cds_by_transcript: transcript_id -> list of CDS features
    """
    genes: dict[str, dict] = {}
    transcripts: dict[str, dict] = {}
    cds_by_transcript: dict[str, list[GFFFeature]] = defaultdict(list)

    for feat in features:
        attrs = feat.attributes
        ftype = feat.type

        if ftype == "gene":
            gene_id = attrs.get("ID")
            if gene_id:
                genes[gene_id] = {
                    "seqid": feat.seqid,
                    "strand": feat.strand,
                    "start": feat.start,
                    "end": feat.end,
                    "attributes": attrs,
                }

        elif ftype in ("mRNA", "transcript"):
            tx_id = attrs.get("ID")
            parents = attrs.get("Parent")
            if not tx_id:
                continue
            if parents:
                gene_id = parents.split(",")[0]
            else:
                gene_id = None

            transcripts[tx_id] = {
                "seqid": feat.seqid,
                "strand": feat.strand,
                "start": feat.start,
                "end": feat.end,
                "gene_id": gene_id,
                "attributes": attrs,
            }

        elif ftype == "CDS":
            parents = attrs.get("Parent")
            if not parents:
                continue
            for tx_id in parents.split(","):
                cds_by_transcript[tx_id].append(feat)

    return genes, transcripts, cds_by_transcript


def load_genome(fasta_path: Path) -> dict[str, SeqIO.SeqRecord]:
    """Load a multi-FASTA genome into a dict keyed by sequence ID."""
    return SeqIO.to_dict(SeqIO.parse(str(fasta_path), "fasta"))


def extract_cds_sequence(
    genome: dict[str, SeqIO.SeqRecord],
    seqid: str,
    cds_features: list["GFFFeature"],
    strand: str,
) -> str:
    """
    Given CDS features for one transcript, pull out the DNA sequence.

    GFF3 coordinates are 1-based inclusive.
    """
    if seqid not in genome:
        raise KeyError(f"Sequence ID {seqid!r} not found in FASTA")

    chrom_seq = genome[seqid].seq
    cds_sorted = sorted(cds_features, key=lambda f: f.start)

    seq_obj = Seq("")
    for cds in cds_sorted:
        # GFF is 1-based inclusive, Python slices are 0-based [start, end)
        piece = chrom_seq[cds.start - 1 : cds.end]
        seq_obj += piece

    if strand == "-":
        seq_obj = seq_obj.reverse_complement()

    return str(seq_obj)


def build_cds_table(
    genome_fasta: Path,
    gff3_path: Path,
):
    """
    Load FASTA and GFF3, and return a DataFrame with one row per transcript.

    Columns:
      - seqid
      - gene_id
      - gene_name
      - transcript_id
      - strand
      - cds_start
      - cds_end
      - cds_length
      - n_cds_exons
      - transcript_attributes (JSON)
      - cds_sequence
    """
    print(f"Loading genome from {genome_fasta} ...")
    genome = load_genome(genome_fasta)

    print(f"Parsing GFF3 from {gff3_path} ...")
    features = read_gff3(gff3_path)

    print("Building gene and transcript maps ...")
    genes, transcripts, cds_by_tx = build_gene_and_transcript_maps(features)

    rows = []
    missing_seqid = set()
    skipped_no_cds = 0
    used = 0

    for tx_id, cds_feats in cds_by_tx.items():
        if not cds_feats:
            skipped_no_cds += 1
            continue

        tx_info = transcripts.get(tx_id)
        if tx_info is None:
            # Some GFF files only list CDS with parent but no explicit mRNA record
            seqid = cds_feats[0].seqid
            strand = cds_feats[0].strand
            gene_id = None
            tx_attrs = {}
        else:
            seqid = tx_info["seqid"]
            strand = tx_info["strand"]
            gene_id = tx_info.get("gene_id")
            tx_attrs = tx_info.get("attributes", {})

        try:
            cds_seq = extract_cds_sequence(
                genome=genome,
                seqid=seqid,
                cds_features=cds_feats,
                strand=strand,
            )
        except KeyError:
            missing_seqid.add(seqid)
            continue

        cds_start = min(f.start for f in cds_feats)
        cds_end = max(f.end for f in cds_feats)

        gene_name = ""
        if gene_id and gene_id in genes:
            gene_attrs = genes[gene_id].get("attributes", {})
            gene_name = gene_attrs.get("Name", "")

        row = {
            "seqid": seqid,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "transcript_id": tx_id,
            "strand": strand,
            "cds_start": cds_start,
            "cds_end": cds_end,
            "cds_length": len(cds_seq),
            "n_cds_exons": len(cds_feats),
            "transcript_attributes": json.dumps(tx_attrs, separators=(",", ":")),
            "cds_sequence": cds_seq,
        }
        rows.append(row)
        used += 1

    print(f"Built rows for {used} transcripts with CDS.")
    if skipped_no_cds:
        print(f"Skipped {skipped_no_cds} transcripts without CDS.")
    if missing_seqid:
        print("Warning: missing sequence IDs in FASTA for:")
        for s in sorted(missing_seqid):
            print("  -", s)

    df = pd.DataFrame(rows)
    df.sort_values(["seqid", "gene_id", "transcript_id", "cds_start"], inplace=True)
    return df


def main():
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"FASTA file not found: {FASTA_PATH}")
    if not GFF3_PATH.exists():
        raise FileNotFoundError(f"GFF3 file not found: {GFF3_PATH}")

    df = build_cds_table(FASTA_PATH, GFF3_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
