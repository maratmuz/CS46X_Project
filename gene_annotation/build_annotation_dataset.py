#!/usr/bin/env python3
"""
Build a strand-aware training dataset from reduced GFF3 + FASTA species folders.

Expected layout:
  <species_root>/<species_id>/.../*.reduced.gff3 (or .reduced.gff)
  <species_root>/<species_id>/.../*.fa|*.fna|*.fasta

For each sampled position, emits one or two samples:
- If sampled position matches target feature on '+' strand, emit '+' sample.
- If sampled position matches target feature on '-' strand, emit '-' sample.
If both match, two rows are emitted.

Feature labels are: Intergenic, UTR, CDS, Intron.
Phase labels are: None for non-CDS, otherwise 0/1/2 for CDS positions.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from Bio import SeqIO
from Bio.Seq import Seq


TRANSCRIPT_TYPES = {"mrna", "transcript"}
UTR_TYPES = {
    "utr",
    "five_prime_utr",
    "three_prime_utr",
    "5utr",
    "3utr",
    "5_prime_utr",
    "3_prime_utr",
}
FEATURE_ORDER = ["Intergenic", "UTR", "CDS", "Intron"]


@dataclass(frozen=True)
class Interval:
    """1-based inclusive interval."""

    start: int
    end: int

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Invalid interval: {self.start}>{self.end}")

    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class CdsSegment:
    start: int
    end: int
    phase: Optional[int]


@dataclass
class SpeciesAnnotations:
    # seqid -> strand -> feature -> merged intervals
    feature_intervals: Dict[str, Dict[str, Dict[str, List[Interval]]]]
    # seqid -> strand -> transcript spans (for intergenic complement)
    transcript_spans: Dict[str, Dict[str, List[Interval]]]
    # seqid -> strand -> raw CDS segments (for phase lookup)
    cds_segments: Dict[str, Dict[str, List[CdsSegment]]]


def open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def parse_gff3_attributes(attr_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def normalize_feature_type(feature_type: str) -> Optional[str]:
    """
    Map raw GFF3 feature type into parser categories.
    - five_prime_UTR / three_prime_UTR (case-insensitive) -> UTR
    """
    ft = feature_type.lower()
    if ft in TRANSCRIPT_TYPES:
        return "TRANSCRIPT"
    if ft == "exon":
        return "EXON"
    if ft == "cds":
        return "CDS"
    if ft in UTR_TYPES:
        return "UTR"
    return None


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: (x.start, x.end))
    out = [sorted_ivs[0]]
    for iv in sorted_ivs[1:]:
        last = out[-1]
        if iv.start <= last.end + 1:
            out[-1] = Interval(last.start, max(last.end, iv.end))
        else:
            out.append(iv)
    return out


def restrict_intervals(intervals: List[Interval], lo: int, hi: int) -> List[Interval]:
    if lo > hi:
        return []
    out: List[Interval] = []
    for iv in intervals:
        s = max(iv.start, lo)
        e = min(iv.end, hi)
        if s <= e:
            out.append(Interval(s, e))
    return out


def complement_intervals(occupied: List[Interval], domain: Interval) -> List[Interval]:
    occ = [iv for iv in occupied if iv.end >= domain.start and iv.start <= domain.end]
    occ = merge_intervals(occ)
    out: List[Interval] = []
    cur = domain.start
    for iv in occ:
        if iv.start > cur:
            out.append(Interval(cur, iv.start - 1))
        cur = max(cur, iv.end + 1)
    if cur <= domain.end:
        out.append(Interval(cur, domain.end))
    return out


def sample_uniform_base(intervals: List[Interval], rng: random.Random) -> Optional[int]:
    if not intervals:
        return None
    lengths = [iv.length() for iv in intervals]
    total = sum(lengths)
    if total <= 0:
        return None
    r = rng.randrange(total)
    acc = 0
    for iv, ln in zip(intervals, lengths):
        if acc + ln > r:
            return iv.start + (r - acc)
        acc += ln
    return None


def total_bases(intervals: List[Interval]) -> int:
    return sum(iv.length() for iv in intervals)


def interval_starts(intervals: List[Interval]) -> List[int]:
    return [iv.start for iv in intervals]


def contains_pos(intervals: List[Interval], starts: List[int], pos: int) -> bool:
    if not intervals:
        return False
    i = bisect.bisect_right(starts, pos) - 1
    if i < 0:
        return False
    iv = intervals[i]
    return iv.start <= pos <= iv.end


def stable_index_name(fasta_path: str) -> str:
    h = hashlib.sha1(fasta_path.encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(fasta_path)}.{h}.sqlite"


def get_fasta_index(species_path: str, fasta_path: str, index_dirname: str):
    idx_dir = os.path.join(species_path, index_dirname)
    os.makedirs(idx_dir, exist_ok=True)
    sqlite_path = os.path.join(idx_dir, stable_index_name(fasta_path))
    return SeqIO.index_db(sqlite_path, [fasta_path], "fasta")


def find_species_dirs(species_root: str) -> List[str]:
    out: List[str] = []
    for ent in sorted(os.scandir(species_root), key=lambda e: e.name):
        if ent.is_dir():
            out.append(ent.path)
    return out


def find_files_with_ext(base_dir: str, exts: Tuple[str, ...]) -> List[str]:
    exts_lower = tuple(x.lower() for x in exts)
    found: List[str] = []
    for root, _, files in os.walk(base_dir):
        for fn in sorted(files):
            fnl = fn.lower()
            for ext in exts_lower:
                if fnl.endswith(ext):
                    found.append(os.path.join(root, fn))
                    break
    return sorted(found)


def choose_species_files(
    species_dir: str,
    fasta_exts: Tuple[str, ...],
    gff_exts: Tuple[str, ...],
) -> Tuple[Optional[str], Optional[str]]:
    gff_files = find_files_with_ext(species_dir, gff_exts)
    fasta_files = find_files_with_ext(species_dir, fasta_exts)

    gff_path = gff_files[0] if gff_files else None
    fasta_path = fasta_files[0] if fasta_files else None

    if len(gff_files) > 1:
        print(f"[WARN] {os.path.basename(species_dir)}: multiple GFFs found, using first: {gff_path}")
    if len(fasta_files) > 1:
        print(f"[WARN] {os.path.basename(species_dir)}: multiple FASTAs found, using first: {fasta_path}")

    return gff_path, fasta_path


def _parse_phase(phase_str: str) -> Optional[int]:
    if phase_str in {".", ""}:
        return None
    try:
        p = int(phase_str)
    except ValueError:
        return None
    if p not in (0, 1, 2):
        return None
    return p


def _flatten_feature_intervals(items: Iterable[Tuple[str, str, Interval]]) -> Tuple[str, str, List[Interval]]:
    items = list(items)
    if not items:
        raise ValueError("Cannot flatten empty feature list")
    seqid = items[0][0]
    strand = items[0][1]
    intervals = [iv for _, _, iv in items]
    return seqid, strand, intervals


def parse_reduced_gff(gff_path: str) -> SpeciesAnnotations:
    transcripts: Dict[str, Tuple[str, str, Interval]] = {}

    exons_by_tx: Dict[str, List[Tuple[str, str, Interval]]] = defaultdict(list)
    utrs_by_tx: Dict[str, List[Tuple[str, str, Interval]]] = defaultdict(list)
    cds_by_tx: Dict[str, List[Tuple[str, str, Interval, Optional[int]]]] = defaultdict(list)

    with open_maybe_gzip(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue

            seqid, _source, feature_type, start_s, end_s, _score, strand, phase_s, attrs_s = cols
            if strand not in {"+", "-"}:
                continue

            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                continue

            attrs = parse_gff3_attributes(attrs_s)
            normalized = normalize_feature_type(feature_type)

            if normalized == "TRANSCRIPT":
                txid = attrs.get("ID") or attrs.get("transcript_id")
                if txid:
                    transcripts[txid] = (seqid, strand, Interval(start, end))
                continue

            if normalized not in {"EXON", "CDS", "UTR"}:
                continue

            parents = [p for p in attrs.get("Parent", "").split(",") if p]
            if not parents:
                continue

            for txid in parents:
                iv = Interval(start, end)
                if normalized == "EXON":
                    exons_by_tx[txid].append((seqid, strand, iv))
                elif normalized == "CDS":
                    cds_by_tx[txid].append((seqid, strand, iv, _parse_phase(phase_s)))
                else:
                    utrs_by_tx[txid].append((seqid, strand, iv))

    # Fallback for files where transcript features are omitted.
    for txid in set(exons_by_tx) | set(utrs_by_tx) | set(cds_by_tx):
        if txid in transcripts:
            continue

        candidates: List[Tuple[str, str, Interval]] = []
        candidates.extend(exons_by_tx.get(txid, []))
        candidates.extend(utrs_by_tx.get(txid, []))
        candidates.extend((a, b, c) for a, b, c, _ in cds_by_tx.get(txid, []))
        if not candidates:
            continue

        seqid, strand, _ = candidates[0]
        starts = [iv.start for _, _, iv in candidates]
        ends = [iv.end for _, _, iv in candidates]
        transcripts[txid] = (seqid, strand, Interval(min(starts), max(ends)))

    feature_intervals: Dict[str, Dict[str, Dict[str, List[Interval]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    transcript_spans: Dict[str, Dict[str, List[Interval]]] = defaultdict(lambda: defaultdict(list))
    cds_segments: Dict[str, Dict[str, List[CdsSegment]]] = defaultdict(lambda: defaultdict(list))

    for txid, (seqid, strand, tx_span) in transcripts.items():
        transcript_spans[seqid][strand].append(tx_span)

        exon_items = exons_by_tx.get(txid, [])
        if exon_items:
            _, _, exon_intervals = _flatten_feature_intervals(exon_items)
            exons_m = merge_intervals(exon_intervals)
            if len(exons_m) >= 2:
                introns: List[Interval] = []
                for left, right in zip(exons_m[:-1], exons_m[1:]):
                    s = left.end + 1
                    e = right.start - 1
                    if s <= e:
                        introns.append(Interval(s, e))
                feature_intervals[seqid][strand]["Intron"].extend(introns)

        utr_items = utrs_by_tx.get(txid, [])
        if utr_items:
            _, _, utr_intervals = _flatten_feature_intervals(utr_items)
            feature_intervals[seqid][strand]["UTR"].extend(utr_intervals)

        cds_items = cds_by_tx.get(txid, [])
        for c_seqid, c_strand, iv, ph in cds_items:
            feature_intervals[c_seqid][c_strand]["CDS"].append(iv)
            cds_segments[c_seqid][c_strand].append(CdsSegment(iv.start, iv.end, ph))

    # Merge all feature intervals + transcript spans.
    for seqid in list(feature_intervals.keys()):
        for strand in list(feature_intervals[seqid].keys()):
            for feature in ["UTR", "CDS", "Intron"]:
                feature_intervals[seqid][strand][feature] = merge_intervals(
                    feature_intervals[seqid][strand].get(feature, [])
                )

    for seqid in list(transcript_spans.keys()):
        for strand in list(transcript_spans[seqid].keys()):
            transcript_spans[seqid][strand] = merge_intervals(transcript_spans[seqid][strand])

    for seqid in list(cds_segments.keys()):
        for strand in list(cds_segments[seqid].keys()):
            cds_segments[seqid][strand] = sorted(
                cds_segments[seqid][strand], key=lambda x: (x.start, x.end)
            )

    return SpeciesAnnotations(
        feature_intervals=feature_intervals,
        transcript_spans=transcript_spans,
        cds_segments=cds_segments,
    )


def phase_for_cds_position(segments: List[CdsSegment], strand: str, pos: int) -> Optional[int]:
    # If multiple CDS segments overlap this position, first sorted match wins.
    for seg in segments:
        if seg.start <= pos <= seg.end:
            if seg.phase is None:
                return None
            first_base_phase = (3 - seg.phase) % 3
            offset = (pos - seg.start) if strand == "+" else (seg.end - pos)
            return (first_base_phase + offset) % 3
    return None


def extract_window(record_seq: Seq, pos1: int, window: int, strand: str) -> Tuple[str, int, int]:
    n = len(record_seq)
    if strand == "+":
        start = pos1 - window + 1
        end = pos1
        if start < 1 or end > n:
            raise ValueError("Out-of-bounds '+' window")
        seq = str(record_seq[start - 1 : end]).upper()
        return seq, start, end

    if strand == "-":
        start = pos1
        end = pos1 + window - 1
        if start < 1 or end > n:
            raise ValueError("Out-of-bounds '-' window")
        seq = str(record_seq[start - 1 : end]).upper()
        return revcomp(seq), start, end

    raise ValueError(f"Unsupported strand: {strand}")


def strand_eligible_domain(contig_len: int, window: int, strand: str) -> Tuple[int, int]:
    if strand == "+":
        return window, contig_len
    if strand == "-":
        return 1, contig_len - window + 1
    raise ValueError(f"Unsupported strand: {strand}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--species_root", required=True,
                   help="Directory containing one subdirectory per species.")
    p.add_argument("--out_tsv", required=True)

    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_intergenic", type=int, default=0)
    p.add_argument("--n_utr", type=int, default=0)
    p.add_argument("--n_cds", type=int, default=0)
    p.add_argument("--n_intron", type=int, default=0)

    p.add_argument("--fasta_exts", default=".fa,.fna,.fasta",
                   help="Comma-separated FASTA extensions (uncompressed for SeqIO.index_db).")
    p.add_argument("--gff_exts", default=".reduced.gff3,.reduced.gff",
                   help="Comma-separated GFF extensions (defaults to reduced GFF files).")
    p.add_argument("--index_dir", default="__fasta_index__")

    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    window = args.window

    if window <= 0:
        raise ValueError("--window must be > 0")

    counts = {
        "Intergenic": args.n_intergenic,
        "UTR": args.n_utr,
        "CDS": args.n_cds,
        "Intron": args.n_intron,
    }

    for k, v in counts.items():
        if v < 0:
            raise ValueError(f"Count for {k} cannot be negative: {v}")

    fasta_exts = tuple(x.strip() for x in args.fasta_exts.split(",") if x.strip())
    gff_exts = tuple(x.strip() for x in args.gff_exts.split(",") if x.strip())

    species_dirs = find_species_dirs(args.species_root)
    if not species_dirs:
        raise ValueError(f"No species subdirectories found under: {args.species_root}")

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)

    out_fields = [
        "species_id",
        "species_path",
        "gff_path",
        "fasta_path",
        "seqname",
        "pos_1based",
        "feature",
        "strand",
        "phase",
        "sequence",
        "window_start_1based",
        "window_end_1based",
        "contig_len",
    ]

    with open(args.out_tsv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()

        for species_path in species_dirs:
            species_id = os.path.basename(species_path.rstrip("/"))
            gff_path, fasta_path = choose_species_files(species_path, fasta_exts, gff_exts)

            if not gff_path:
                print(f"[WARN] {species_id}: no GFF found with extensions {gff_exts}; skipping")
                continue
            if not fasta_path:
                print(f"[WARN] {species_id}: no FASTA found with extensions {fasta_exts}; skipping")
                continue
            if fasta_path.endswith(".gz"):
                print(f"[WARN] {species_id}: FASTA is gzipped ({fasta_path}); skipping")
                continue

            print(f"[INFO] {species_id}: parsing {gff_path}")
            ann = parse_reduced_gff(gff_path)

            idx = get_fasta_index(species_path, fasta_path, args.index_dir)
            try:
                # Build intergenic intervals per seqid/strand as complement of transcript spans.
                for seqname in idx.keys():
                    contig_len = len(idx[seqname].seq)
                    domain = Interval(1, contig_len)
                    for strand in ["+", "-"]:
                        tx_spans = ann.transcript_spans.get(seqname, {}).get(strand, [])
                        intergenic = complement_intervals(tx_spans, domain)
                        ann.feature_intervals[seqname][strand]["Intergenic"] = intergenic

                for feature in FEATURE_ORDER:
                    n_target = counts[feature]
                    if n_target <= 0:
                        continue

                    contigs: List[Dict[str, object]] = []
                    weights: List[int] = []

                    for seqname in idx.keys():
                        record = idx[seqname]
                        contig_len = len(record.seq)

                        plus_raw = ann.feature_intervals.get(seqname, {}).get("+", {}).get(feature, [])
                        minus_raw = ann.feature_intervals.get(seqname, {}).get("-", {}).get(feature, [])

                        lo_p, hi_p = strand_eligible_domain(contig_len, window, "+")
                        lo_m, hi_m = strand_eligible_domain(contig_len, window, "-")

                        plus_eligible = restrict_intervals(plus_raw, lo_p, hi_p)
                        minus_eligible = restrict_intervals(minus_raw, lo_m, hi_m)

                        union_eligible = merge_intervals(plus_eligible + minus_eligible)
                        union_bases = total_bases(union_eligible)
                        if union_bases <= 0:
                            continue

                        contigs.append(
                            {
                                "seqname": seqname,
                                "contig_len": contig_len,
                                "plus_intervals": plus_eligible,
                                "minus_intervals": minus_eligible,
                                "plus_starts": interval_starts(plus_eligible),
                                "minus_starts": interval_starts(minus_eligible),
                                "union_intervals": union_eligible,
                            }
                        )
                        weights.append(union_bases)

                    if not contigs:
                        print(f"[WARN] {species_id}: no eligible bases for feature={feature}")
                        continue

                    print(f"[INFO] {species_id}: sampling feature={feature}, positions={n_target}")

                    for _ in range(n_target):
                        total_w = sum(weights)
                        pick = rng.randrange(total_w)
                        acc = 0
                        chosen = contigs[-1]
                        for c, w in zip(contigs, weights):
                            if acc + w > pick:
                                chosen = c
                                break
                            acc += w

                        seqname = chosen["seqname"]
                        contig_len = chosen["contig_len"]
                        plus_intervals = chosen["plus_intervals"]
                        minus_intervals = chosen["minus_intervals"]
                        plus_starts = chosen["plus_starts"]
                        minus_starts = chosen["minus_starts"]
                        union_intervals = chosen["union_intervals"]

                        pos = sample_uniform_base(union_intervals, rng)
                        if pos is None:
                            continue

                        record = idx[seqname]

                        for strand, intervals, starts in [
                            ("+", plus_intervals, plus_starts),
                            ("-", minus_intervals, minus_starts),
                        ]:
                            if not contains_pos(intervals, starts, pos):
                                continue

                            try:
                                seq, ws, we = extract_window(record.seq, pos, window, strand)
                            except ValueError:
                                continue

                            if feature == "CDS":
                                phase_val = phase_for_cds_position(
                                    ann.cds_segments.get(seqname, {}).get(strand, []),
                                    strand,
                                    pos,
                                )
                                phase_out = str(phase_val) if phase_val is not None else "None"
                            else:
                                phase_out = "None"

                            writer.writerow(
                                {
                                    "species_id": species_id,
                                    "species_path": species_path,
                                    "gff_path": gff_path,
                                    "fasta_path": fasta_path,
                                    "seqname": seqname,
                                    "pos_1based": pos,
                                    "feature": feature,
                                    "strand": strand,
                                    "phase": phase_out,
                                    "sequence": seq,
                                    "window_start_1based": ws,
                                    "window_end_1based": we,
                                    "contig_len": contig_len,
                                }
                            )
            finally:
                idx.close()

    print(f"Done. Wrote: {args.out_tsv}")


if __name__ == "__main__":
    main()
