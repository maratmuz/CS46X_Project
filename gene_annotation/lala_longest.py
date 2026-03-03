#! /usr/bin/env python

"""
Usage:
    python myscript.py [options]

Warnings:
-   Normal print statements here will be funneled into the reduced gff3 files. 
    If prints are needed for debugging, print to stderr.

"""

import click
from dustdas import gffhelper


def gff_gen(gff_file):
    reader = gffhelper.read_gff_file(gff_file)
    for entry in reader:
        clean_entry(entry)
        yield entry

def clean_entry(entry):
    # always present and integers
    entry.start = int(entry.start)
    entry.end = int(entry.end)


def str_entry(entry):
    out = [entry.seqid,
           entry.source,
           entry.type,
           entry.start,
           entry.end,
           entry.score,
           entry.strand,
           entry.phase,
           entry.attribute]
    return "\t".join([str(c) for c in out])


def cds_length(entries):
    l = 0
    for entry in entries:
        if entry.type == "CDS":
            l += entry.end - entry.start + 1
    return l


def parse_attributes(attribute_field):
    return dict(
        item.split("=", 1)
        for item in attribute_field.split(";")
        if "=" in item
    )


def is_protein_coding_gene(attribute_field):
    attr_dict = parse_attributes(attribute_field)
    biotype = (
        attr_dict.get("gene_biotype")
        or attr_dict.get("biotype")
        or attr_dict.get("gene_type")
    )
    return biotype == "protein_coding"


def emit_longest(gene_entry, transcripts):
    if not transcripts:
        return
    longest = max(transcripts.values(), key=lambda x: cds_length(x))
    print(str_entry(gene_entry))
    for e in longest:
        print(str_entry(e))


@click.command()
@click.option("-i", "--gff-file", required=True)
def main(gff_file):
    transcripts = {}
    keep_gene = False
    gene_entry = None
    gene_id = None
    transcript_id = None
    for entry in gff_gen(gff_file):
        if entry.type in ["gene", "pseudogene"]:
            if keep_gene and gene_entry is not None:
                emit_longest(gene_entry, transcripts)

            transcripts = {}
            transcript_id = None
            gene_id = entry.get_ID()
            gene_entry = entry
            keep_gene = is_protein_coding_gene(entry.attribute)
        if not keep_gene:
            continue
        elif entry.type in ['mRNA', 'transcript']:
            assert entry.get_Parent()[0] == gene_id, f"{entry.get_Parent()[0]} != {gene_id}"
            transcript_id = entry.get_ID()
            transcripts[transcript_id] = [entry]
        elif entry.type.lower() in {
            "exon",
            "cds",
            "utr",
            "five_prime_utr",
            "three_prime_utr",
            "5utr",
            "3utr",
            "5_prime_utr",
            "3_prime_utr",
        }:
            # assumes sorted order, checking would be safer
            
            # this throws when a gene didn't have an mRNA/transcript feature
            # but skipped straight to exon/CDS
            # will just skip to the next gene
            # it's mostly because tRNA...
            try:
                transcripts[transcript_id].append(entry)
            except KeyError:
                continue

    if keep_gene and gene_entry is not None:
        emit_longest(gene_entry, transcripts)

if __name__ == "__main__":
    main()
