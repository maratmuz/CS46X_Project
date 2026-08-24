import contextlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_supplemental_loci import (
    ANCHOR_OFFSET,
    SPECIES,
    Feature,
    Interval,
    NearestGeneIndex,
    PgbRecord,
    SequenceClassification,
    SpeciesInputs,
    audit_pgb,
    balanced_quotas,
    calculate_intergenic_availability,
    classify_reference_sequences,
    deduplicate_pseudogenes,
    deterministic_cap,
    discover_species_inputs,
    intergenic_records,
    parse_attributes,
    prepare_pseudogene_candidates,
    pseudogene_record,
    pseudogene_rule,
    process_species,
    reverse_complement,
    sample_intergenic_candidates,
)


class FakeFasta:
    def __init__(self, sequences, descriptions=None):
        self.sequences = {key: value.upper() for key, value in sequences.items()}
        self.descriptions = descriptions or {}
        self.keys = tuple(self.sequences)

    def resolve(self, seqid):
        if seqid in self.sequences:
            return seqid
        without_version = seqid.rsplit(".", 1)[0]
        matches = [key for key in self.keys if key.rsplit(".", 1)[0] == without_version]
        return matches[0] if len(matches) == 1 else None

    def length(self, seqid):
        return len(self.sequences[self.resolve(seqid)])

    def fetch(self, seqid, start, end):
        return self.sequences[self.resolve(seqid)][start:end]

    def description(self, seqid):
        resolved = self.resolve(seqid)
        return self.descriptions.get(resolved, resolved)

    def close(self):
        pass


def feature(
    feature_type="gene",
    start=6_000,
    end=7_000,
    strand="+",
    raw="ID=gene1",
    seqid="chr1",
    line=1,
):
    return Feature(
        seqid,
        "NCBI",
        feature_type,
        start,
        end,
        strand,
        raw,
        parse_attributes(raw),
        line,
    )


class AttributeAndPseudogeneTests(unittest.TestCase):
    def test_gff_and_gtf_attributes(self):
        gff = parse_attributes("ID=gene-1;Dbxref=GeneID:123,RefSeq:ABC;Note=a%20note;pseudo")
        self.assertEqual(gff["Dbxref"], ["GeneID:123", "RefSeq:ABC"])
        self.assertEqual(gff["Note"], ["a note"])
        self.assertEqual(gff["pseudo"], ["true"])
        gtf = parse_attributes('gene_id "AT1G01010"; gene_biotype "pseudogene";')
        self.assertEqual(gtf["gene_id"], ["AT1G01010"])

    def test_primary_fallback_and_children(self):
        explicit = feature("pseudogene")
        fallback = feature("gene", raw="ID=g2;gene_biotype=unitary_pseudogene")
        bare_pseudo = feature("gene", raw="ID=g3;pseudo=true")
        child = feature("pseudogene", raw="ID=child;Parent=g1")
        transcript = feature("mRNA", raw="ID=rna1;gene_biotype=pseudogene")
        self.assertEqual(pseudogene_rule(explicit), (True, True))
        self.assertEqual(pseudogene_rule(fallback), (True, False))
        self.assertEqual(pseudogene_rule(bare_pseudo), (True, False))
        self.assertEqual(pseudogene_rule(child), (False, False))
        self.assertEqual(pseudogene_rule(transcript), (False, False))

    def test_deduplication_prefers_explicit_feature(self):
        fallback = feature(
            "gene", raw="ID=gene-version;gene_biotype=pseudogene;locus_tag=LOC1", line=1
        )
        fallback.explicit_pseudogene = False
        explicit = feature("pseudogene", raw="ID=pseudo-version;locus_tag=LOC1", line=2)
        explicit.explicit_pseudogene = True
        result = deduplicate_pseudogenes([fallback, explicit])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].feature_type, "pseudogene")


class StrandAndBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.sequence = ("ACGTTGCA" * 3_000)[:20_000]
        self.fasta = FakeFasta({"chr1": self.sequence})
        self.classifications = {
            "chr1": SequenceClassification("nuclear", "test", primary_chromosome=True)
        }
        self.spec = SPECIES[0]

    def test_plus_anchor_is_at_offset_5000(self):
        item = feature("pseudogene", start=6_000, end=7_000, strand="+", raw="ID=plus")
        status = prepare_pseudogene_candidates(
            [item], self.fasta, self.classifications, False
        )[0]
        record = pseudogene_record(self.spec, status, self.fasta, False)
        self.assertEqual(len(record.sequence), 6_000)
        self.assertEqual(record.sequence[ANCHOR_OFFSET], self.sequence[item.start])
        self.assertEqual(record.start, 1_000)
        self.assertEqual(record.end, 7_000)

    def test_minus_anchor_and_reverse_complement(self):
        item = feature("pseudogene", start=8_000, end=10_000, strand="-", raw="ID=minus")
        status = prepare_pseudogene_candidates(
            [item], self.fasta, self.classifications, False
        )[0]
        record = pseudogene_record(self.spec, status, self.fasta, False)
        self.assertEqual(record.sequence, reverse_complement(self.sequence[9_000:15_000]))
        self.assertEqual(
            record.sequence[ANCHOR_OFFSET], reverse_complement(self.sequence[item.end - 1])[0]
        )
        self.assertEqual(record.anchor_genomic_1based, item.end)

    def test_boundary_exclusion_and_padding(self):
        item = feature("pseudogene", start=1_000, end=2_000, strand="+", raw="ID=edge")
        excluded = prepare_pseudogene_candidates(
            [item], self.fasta, self.classifications, False
        )[0]
        self.assertFalse(excluded.eligible)
        self.assertEqual(excluded.exclusion_reason, "boundary_truncated")
        padded = prepare_pseudogene_candidates(
            [item], self.fasta, self.classifications, True
        )[0]
        record = pseudogene_record(self.spec, padded, self.fasta, True)
        self.assertEqual(record.sequence[:4_000], "N" * 4_000)
        self.assertEqual(record.sequence[ANCHOR_OFFSET], self.sequence[item.start])
        self.assertEqual(len(record.sequence), 6_000)


class ClassificationTests(unittest.TestCase):
    def test_precedence_and_nc_prefix(self):
        fasta = FakeFasta(
            {
                "NC_1": "A" * 100,
                "cp": "A" * 100,
                "mt": "A" * 100,
                "chr1": "A" * 100,
            },
            {
                "NC_1": "NC_1 unplaced genomic scaffold",
                "cp": "cp chromosome 1",  # report must override this nuclear-looking text
                "mt": "complete mitochondrial genome",
                "chr1": "chromosome 1",
            },
        )
        reports = {
            "cp": SequenceClassification("chloroplast", "assembly_report", "assembled-molecule")
        }
        result = classify_reference_sequences(fasta, reports, {})
        self.assertEqual(result["NC_1"].compartment, "unknown")
        self.assertEqual(result["cp"].compartment, "chloroplast")
        self.assertEqual(result["mt"].compartment, "mitochondrial")
        self.assertEqual(result["chr1"].compartment, "nuclear")

    def test_pgb_identifier_and_exact_sequence_mapping(self):
        sequence = ("ACGT" * 5_000)[:20_000]
        fasta = FakeFasta({"chr1": sequence, "cp": "C" * 20_000})
        nuclear_gene = feature(
            "gene", start=6_000, end=7_000, raw="ID=gene:NUC1;Dbxref=GeneID:100", seqid="chr1"
        )
        chloroplast_gene = feature(
            "gene", start=6_000, end=7_000, raw="ID=CP1;locus_tag=CP_TAG", seqid="cp"
        )
        classifications = {
            "chr1": SequenceClassification("nuclear", "test"),
            "cp": SequenceClassification("chloroplast", "test"),
        }
        exact_sequence = sequence[1_000:7_000]
        mappings, _ = audit_pgb(
            [
                PgbRecord("100", "A" * 6_000),
                PgbRecord("CP_TAG", "A" * 6_000),
                PgbRecord("no_matching_id", exact_sequence),
            ],
            [nuclear_gene, chloroplast_gene],
            fasta,
            classifications,
            True,
        )
        self.assertTrue(mappings[0].mapped)
        self.assertEqual(mappings[0].compartment, "nuclear")
        self.assertTrue(mappings[1].mapped)
        self.assertEqual(mappings[1].compartment, "chloroplast")
        self.assertTrue(mappings[2].mapped)
        self.assertEqual(mappings[2].evidence, "exact_oriented_6000bp_sequence")


class MaskingAndSamplingTests(unittest.TestCase):
    def setUp(self):
        self.fasta = FakeFasta(
            {
                "chr1": ("ACGT" * 7_500),
                "chr2": ("GCGT" * 7_500),
                "unknown": "N" * 30_000,
            }
        )
        self.classifications = {
            "chr1": SequenceClassification("nuclear", "test", primary_chromosome=True),
            "chr2": SequenceClassification("nuclear", "test", primary_chromosome=True),
            "unknown": SequenceClassification("unknown", "test"),
        }

    def test_mask_complement_and_capacity(self):
        intervals, total_nt, capacity, remainder = calculate_intergenic_availability(
            self.fasta,
            self.classifications,
            {"chr1": [Interval(5_000, 10_000)], "chr2": [Interval(0, 30_000)]},
            False,
            False,
        )
        self.assertEqual(intervals["chr1"], [Interval(0, 5_000), Interval(10_000, 30_000)])
        self.assertEqual(total_nt, 25_000)
        self.assertEqual(capacity, 3)
        self.assertEqual(remainder, 7_000)
        self.assertNotIn("unknown", intervals)

    def test_deterministic_nonoverlapping_sampling(self):
        intervals = {"chr1": [Interval(0, 30_000)], "chr2": [Interval(0, 30_000)]}
        first = sample_intergenic_candidates(
            SPECIES[0], self.fasta, self.classifications, intervals, 6, 17, []
        )
        second = sample_intergenic_candidates(
            SPECIES[0], self.fasta, self.classifications, intervals, 6, 17, []
        )
        self.assertEqual(
            [(item.seqid, item.start, item.sequence) for item in first],
            [(item.seqid, item.start, item.sequence) for item in second],
        )
        self.assertEqual(len(first), 6)
        for left_index, left in enumerate(first):
            self.assertEqual(left.end - left.start, 6_000)
            for right in first[left_index + 1 :]:
                if left.seqid == right.seqid:
                    self.assertTrue(left.end <= right.start or right.end <= left.start)

    def test_balancing_prefers_primary_chromosomes(self):
        quotas = balanced_quotas(
            6, {"chr1": 10, "chr2": 10, "scaffold": 100}, {"chr1", "chr2"}
        )
        self.assertEqual(quotas, {"chr1": 3, "chr2": 3, "scaffold": 0})

    def test_cap_is_repeatable(self):
        values = list(range(100))
        self.assertEqual(
            deterministic_cap(values, 10, 99, "species", "pseudo"),
            deterministic_cap(values, 10, 99, "species", "pseudo"),
        )
        self.assertNotEqual(
            deterministic_cap(values, 10, 99, "species", "pseudo"),
            deterministic_cap(values, 10, 100, "species", "pseudo"),
        )

    def test_nearest_gene_metadata(self):
        before = feature("gene", start=0, end=2_000, raw="ID=before")
        after = feature("gene", start=20_000, end=22_000, raw="ID=after")
        candidate = sample_intergenic_candidates(
            SPECIES[0],
            self.fasta,
            self.classifications,
            {"chr1": [Interval(10_000, 16_000)]},
            1,
            2,
            [],
        )
        rows = intergenic_records(SPECIES[0], candidate, NearestGeneIndex([before, after]))
        self.assertEqual(rows[0].extra["nearest_gene_id"], "after")
        self.assertEqual(rows[0].extra["nearest_gene_distance_bp"], 4_000)


class DiscoveryTests(unittest.TestCase):
    def test_resolves_only_required_species_using_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for spec in SPECIES:
                directory = root / f"assembly_{spec.taxid}"
                directory.mkdir()
                (directory / "genomic.fna").write_text(">chr1\nAAAA\n", encoding="utf-8")
                (directory / "genomic.gff").write_text("##gff-version 3\n", encoding="utf-8")
                (directory / "assembly_report.txt").write_text(
                    f"# Organism name: {spec.scientific_name}\n# Taxid: {spec.taxid}\n",
                    encoding="utf-8",
                )
            ignored = root / "human"
            ignored.mkdir()
            (ignored / "genomic.fna").write_text(">chr1\nAAAA\n", encoding="utf-8")
            (ignored / "genomic.gff").write_text("##gff-version 3\n", encoding="utf-8")
            resolved = discover_species_inputs(root)
            self.assertEqual(set(resolved), {spec.slug for spec in SPECIES})
            for spec in SPECIES:
                self.assertEqual(resolved[spec.slug].directory.name, f"assembly_{spec.taxid}")


class ProcessSpeciesSmokeTest(unittest.TestCase):
    def test_writes_required_species_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            fasta_path = source / "genomic.fna"
            annotation_path = source / "genomic.gff"
            pgb_path = root / "arabidopsis_thaliana.fasta"
            fasta_path.write_text(
                ">chr1 chromosome 1 nuclear\n" + "A" * 30_000 + "\n", encoding="utf-8"
            )
            annotation_path.write_text(
                "\n".join(
                    (
                        "##gff-version 3",
                        "chr1\tNCBI\tregion\t1\t30000\t.\t+\t.\tID=region1;chromosome=1",
                        "chr1\tNCBI\tgene\t7001\t8000\t.\t+\t.\tID=gene1",
                        "chr1\tNCBI\tpseudogene\t7001\t8000\t.\t+\t.\tID=pseudo1;Name=PSEUDO",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            pgb_path.write_text(">gene1|low\n" + "A" * 6_000 + "\n", encoding="utf-8")
            inputs = SpeciesInputs(
                SPECIES[0], source, fasta_path, annotation_path, (), ("test fixture",)
            )
            output = root / "output"
            output.mkdir()
            args = type(
                "Args",
                (),
                {
                    "gene_buffer": 0,
                    "exact_sequence_mapping": False,
                    "pad_boundaries": False,
                    "max_pseudogenes": 5,
                    "max_intergenic": 2,
                    "seed": 11,
                    "include_organelles": False,
                    "include_unknown": False,
                    "force": False,
                },
            )()
            fake = FakeFasta(
                {"chr1": "A" * 30_000}, {"chr1": "chr1 chromosome 1 nuclear"}
            )
            if importlib.util.find_spec("pyfaidx") is None:
                fasta_context = patch(
                    "scripts.build_supplemental_loci.IndexedFasta", return_value=fake
                )
            else:
                fasta_context = contextlib.nullcontext()
            with fasta_context:
                result = process_species(inputs, [pgb_path], output, args)
            species_output = output / SPECIES[0].slug
            for name in (
                "nuclear_pseudogenes.fasta",
                "nuclear_pseudogenes.tsv",
                "organellar_pseudogenes.tsv",
                "unknown_pseudogenes.tsv",
                "nuclear_intergenic.fasta",
                "nuclear_intergenic.tsv",
                "pgb_compartment_audit.tsv",
                "excluded.tsv",
                "qc.json",
            ):
                self.assertTrue((species_output / name).is_file(), name)
            self.assertEqual(result.summary["nuclear_pseudogenes_emitted"], 1)
            self.assertEqual(result.summary["nuclear_intergenic_emitted"], 2)
            self.assertEqual(result.summary["pgb_mapping_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
