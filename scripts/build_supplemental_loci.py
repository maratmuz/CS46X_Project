#!/usr/bin/env python3
"""Build pseudogene and intergenic supplements for AgroNT's PGB benchmark.

The reference FASTA is accessed through pyfaidx and is never loaded wholesale.
Annotations and small report files are streamed once per species.  The module is
also intentionally importable so that coordinate and sampling behavior can be
tested with an in-memory FASTA stand-in.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import random
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TextIO, TypeVar
from urllib.parse import unquote


WINDOW_LENGTH = 6_000
ANCHOR_OFFSET = 5_000
UPSTREAM_LENGTH = 5_000
DOWNSTREAM_LENGTH = 1_000
SCRIPT_VERSION = "1.0"
T = TypeVar("T")


@dataclass(frozen=True)
class SpeciesSpec:
    scientific_name: str
    slug: str
    taxid: str
    aliases: tuple[str, ...]


SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec(
        "Arabidopsis thaliana",
        "arabidopsis_thaliana",
        "3702",
        ("arabidopsis", "a_thaliana", "athaliana", "arath"),
    ),
    SpeciesSpec(
        "Glycine max",
        "glycine_max",
        "3847",
        ("glycine", "soybean", "g_max", "gmax"),
    ),
    SpeciesSpec(
        "Oryza sativa",
        "oryza_sativa",
        "4530",
        ("oryza", "rice", "o_sativa", "osativa"),
    ),
    SpeciesSpec(
        "Solanum lycopersicum",
        "solanum_lycopersicum",
        "4081",
        ("solanum", "tomato", "s_lycopersicum", "slycopersicum"),
    ),
    SpeciesSpec(
        "Zea mays",
        "zea_mays",
        "4577",
        ("zea", "maize", "z_mays", "zmays"),
    ),
)


class BuildError(RuntimeError):
    """A clear, user-actionable build failure."""


@dataclass(frozen=True)
class SpeciesInputs:
    spec: SpeciesSpec
    directory: Path
    fasta: Path
    annotation: Path
    reports: tuple[Path, ...]
    resolution_evidence: tuple[str, ...]


@dataclass
class Feature:
    seqid: str
    source: str
    feature_type: str
    start: int
    end: int
    strand: str
    attributes_raw: str
    attributes: dict[str, list[str]]
    line_number: int
    explicit_pseudogene: bool = False

    @property
    def coordinate_key(self) -> tuple[str, int, int, str]:
        return (self.seqid, self.start, self.end, self.strand)


@dataclass(frozen=True)
class SequenceClassification:
    compartment: str
    evidence: str
    sequence_role: str = ""
    molecule: str = ""
    primary_chromosome: bool = False


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def capacity(self) -> int:
        return self.length // WINDOW_LENGTH


@dataclass
class WindowRecord:
    dataset_id: str
    species: str
    locus_category: str
    seqid: str
    start: int
    end: int
    strand: str
    compartment: str
    anchor_type: str
    anchor_genomic_1based: int
    anchor_offset_0based: int
    gc_fraction: float
    n_fraction: float
    sequence: str
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateStatus:
    feature: Feature
    compartment: str
    classification_evidence: str
    eligible: bool
    exclusion_reason: str = ""
    requested_start: int | None = None
    requested_end: int | None = None


@dataclass
class PgbRecord:
    gene_id: str
    sequence: str
    source_files: set[str] = field(default_factory=set)
    conflicting_sequences: bool = False


@dataclass
class PgbMapping:
    gene_id: str
    mapped: bool
    compartment: str
    evidence: str
    annotation_id: str = ""
    seqid: str = ""
    note: str = ""


@dataclass
class SpeciesResult:
    spec: SpeciesSpec
    inputs: SpeciesInputs
    summary: dict[str, object]
    pseudogene_records: list[WindowRecord]
    intergenic_records: list[WindowRecord]
    pgb_mappings: list[PgbMapping]
    exclusions: list[dict[str, object]]
    qc: dict[str, object]


def open_text(path: Path) -> TextIO:
    """Open a plain or gzip-compressed text file."""

    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "unnamed"


def first_attribute(feature: Feature, *names: str) -> str:
    wanted = {name.casefold() for name in names}
    for key, values in feature.attributes.items():
        if key.casefold() in wanted:
            for value in values:
                if value:
                    return value
    return ""


def attribute_values(feature: Feature, *names: str) -> list[str]:
    wanted = {name.casefold() for name in names}
    result: list[str] = []
    for key, values in feature.attributes.items():
        if key.casefold() in wanted:
            result.extend(value for value in values if value)
    return result


def parse_attributes(raw: str) -> dict[str, list[str]]:
    """Parse either GFF3 key=value or GTF key "value" attributes."""

    parsed: dict[str, list[str]] = defaultdict(list)
    for item in raw.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            values = value.split(",") if key.casefold() not in {"note", "description"} else [value]
        else:
            match = re.match(r"(\S+)\s+[\"']?(.*?)[\"']?$", item)
            if match:
                key, value = match.groups()
                values = [value]
            else:
                key, values = item, ["true"]
        key = unquote(key.strip())
        for value in values:
            parsed[key].append(unquote(value.strip().strip('"').strip("'")))
    return dict(parsed)


def iter_gff(path: Path) -> Iterator[Feature]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n\r").split("\t")
            if len(columns) != 9:
                continue
            try:
                start_1based = int(columns[3])
                end_1based = int(columns[4])
            except ValueError:
                continue
            if start_1based < 1 or end_1based < start_1based:
                continue
            yield Feature(
                seqid=columns[0],
                source=columns[1],
                feature_type=columns[2],
                start=start_1based - 1,
                end=end_1based,
                strand=columns[6],
                attributes_raw=columns[8],
                attributes=parse_attributes(columns[8]),
                line_number=line_number,
            )


def is_top_level(feature: Feature) -> bool:
    return not attribute_values(feature, "Parent", "parent")


def pseudogene_rule(feature: Feature) -> tuple[bool, bool]:
    """Return (is_pseudogene, is_explicit_pseudogene_feature)."""

    if not is_top_level(feature):
        return False, False
    feature_type = feature.feature_type.casefold()
    if feature_type == "pseudogene":
        return True, True
    if feature_type != "gene":
        return False, False
    biotypes = attribute_values(feature, "gene_biotype", "gene_type", "biotype")
    pseudo = first_attribute(feature, "pseudo")
    fallback = any("pseudogene" in value.casefold() for value in biotypes)
    fallback = fallback or pseudo.casefold() == "true"
    return fallback, False


def is_mask_feature(feature: Feature) -> bool:
    kind = feature.feature_type.casefold().replace("-", "_")
    if kind in {"cds", "exon"}:
        return True
    return "gene" in kind or "rna" in kind or "transcript" in kind


def is_nearest_gene_feature(feature: Feature) -> bool:
    kind = feature.feature_type.casefold().replace("-", "_")
    return is_top_level(feature) and (kind == "gene" or "gene" in kind)


def stable_identifiers(feature: Feature) -> list[tuple[str, str]]:
    """Return identifiers with the evidence type used for PGB mapping."""

    result: list[tuple[str, str]] = []
    direct = (
        ("stable_id", ("ID", "gene_id")),
        ("locus_tag", ("locus_tag", "old_locus_tag")),
        ("gene_name", ("Name", "gene", "gene_name")),
        ("alias", ("Alias",)),
    )
    for evidence, keys in direct:
        for value in attribute_values(feature, *keys):
            result.append((value, evidence))
    for raw_xref in attribute_values(feature, "Dbxref", "db_xref", "xref"):
        for xref in raw_xref.split(","):
            xref = xref.strip()
            if not xref:
                continue
            result.append((xref, "cross_reference"))
            if ":" in xref:
                namespace, value = xref.split(":", 1)
                evidence = "NCBI_GeneID" if namespace.casefold() == "geneid" else "cross_reference"
                result.append((value, evidence))
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value, evidence in result:
        pair = (value.strip(), evidence)
        if pair[0] and pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def canonical_feature_id(feature: Feature) -> str:
    return (
        first_attribute(feature, "ID", "gene_id", "locus_tag", "Name", "gene_name")
        or next((value for value, evidence in stable_identifiers(feature) if evidence == "NCBI_GeneID"), "")
        or f"{feature.seqid}_{feature.start + 1}_{feature.end}_{feature.strand}"
    )


def deduplicate_pseudogenes(features: Sequence[Feature]) -> list[Feature]:
    """Union duplicate identities, preferring explicit pseudogene records."""

    if not features:
        return []
    parents = list(range(len(features)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen_keys: dict[tuple[str, str], int] = {}
    for index, feature in enumerate(features):
        keys: list[tuple[str, str]] = []
        for key_name, attr_names in (
            ("stable_id", ("ID", "gene_id")),
            ("locus_tag", ("locus_tag",)),
        ):
            keys.extend((key_name, value.casefold()) for value in attribute_values(feature, *attr_names))
        keys.extend(
            ("geneid", value.casefold())
            for value, evidence in stable_identifiers(feature)
            if evidence == "NCBI_GeneID"
        )
        keys.append(("coordinates", repr(feature.coordinate_key)))
        for key in keys:
            if key in seen_keys:
                union(index, seen_keys[key])
            else:
                seen_keys[key] = index

    groups: dict[int, list[Feature]] = defaultdict(list)
    for index, feature in enumerate(features):
        groups[find(index)].append(feature)

    def preference(feature: Feature) -> tuple[int, int, int, str]:
        return (
            int(feature.feature_type.casefold() == "pseudogene"),
            len(stable_identifiers(feature)),
            len(feature.attributes_raw),
            canonical_feature_id(feature),
        )

    selected = [max(group, key=preference) for group in groups.values()]
    return sorted(selected, key=lambda item: (item.seqid, item.start, item.end, item.strand, canonical_feature_id(item)))


def choose_input_file(files: Sequence[Path], kind: str, directory: Path) -> Path:
    if not files:
        raise BuildError(f"No {kind} file found under resolved species directory: {directory}")

    def rank(path: Path) -> tuple[int, int, str]:
        name = path.name.casefold()
        if kind == "genomic FASTA":
            bad = any(token in name for token in ("cds", "rna", "protein", "pep", "transcript"))
            score = 0 if bad else 20
            if "genomic" in name:
                score += 20
            if name in {"genomic.fna", "genomic.fna.gz"}:
                score += 10
        else:
            score = 20 if "genomic" in name else 0
            if name in {
                "genomic.gff",
                "genomic.gff3",
                "genomic.gtf",
                "genomic.gff.gz",
                "genomic.gff3.gz",
                "genomic.gtf.gz",
            }:
                score += 10
            # Prefer GFF/GFF3 annotations when both NCBI GFF and GTF exports
            # are present. GTF remains a supported fallback.
            if re.search(r"\.gff3?(?:\.gz)?$", name):
                score += 20
        return score, -len(path.parts), name

    ranked = sorted(files, key=rank, reverse=True)
    top_rank = rank(ranked[0])[:2]
    tied = [path for path in ranked if rank(path)[:2] == top_rank]
    if len(tied) > 1:
        choices = "\n  - ".join(str(path) for path in tied)
        raise BuildError(f"Ambiguous {kind} files in {directory}:\n  - {choices}")
    return ranked[0]


def _small_metadata_text(paths: Iterable[Path], limit_per_file: int = 2_000_000) -> str:
    chunks: list[str] = []
    for path in paths:
        try:
            with open_text(path) as handle:
                chunks.append(handle.read(limit_per_file))
        except (OSError, UnicodeError):
            continue
    return "\n".join(chunks)


def species_match_evidence(
    spec: SpeciesSpec, directory: Path, metadata_text: str
) -> tuple[int, tuple[str, ...]]:
    metadata_norm = normalized_text(metadata_text)
    path_norm = normalized_text(str(directory))
    scientific_norm = normalized_text(spec.scientific_name)
    evidence: list[str] = []
    rank = 0
    if scientific_norm in metadata_norm:
        rank = max(rank, 100)
        evidence.append(f"metadata scientific name: {spec.scientific_name}")
    taxid_patterns = (
        rf"tax(?:onomy)?[_ -]?id[^0-9]{{0,8}}{re.escape(spec.taxid)}(?:\D|$)",
        rf"species[_ -]?taxid[^0-9]{{0,8}}{re.escape(spec.taxid)}(?:\D|$)",
    )
    if any(re.search(pattern, metadata_text, re.IGNORECASE) for pattern in taxid_patterns):
        rank = max(rank, 95)
        evidence.append(f"metadata NCBI taxonomy ID: {spec.taxid}")
    if scientific_norm in path_norm:
        rank = max(rank, 70)
        evidence.append(f"directory scientific name: {spec.scientific_name}")
    for alias in spec.aliases:
        alias_norm = normalized_text(alias)
        if len(alias_norm) >= 4 and re.search(rf"(?:^|_){re.escape(alias_norm)}(?:_|$)", path_norm):
            rank = max(rank, 50)
            evidence.append(f"directory alias: {alias}")
            break
    return rank, tuple(evidence)


def discover_species_inputs(genomes_root: Path) -> dict[str, SpeciesInputs]:
    if not genomes_root.is_dir():
        symlink = next(
            (path for path in (genomes_root, *genomes_root.parents) if path.is_symlink()),
            None,
        )
        suffix = ""
        if symlink is not None:
            suffix = f" (symlink {symlink} -> {os.readlink(symlink)} is unavailable on this host)"
        raise BuildError(f"Genomes root is not an accessible directory: {genomes_root}{suffix}")

    candidate_dirs: list[tuple[Path, list[Path], list[Path], list[Path], str]] = []
    for current, directories, filenames in os.walk(genomes_root):
        directories[:] = sorted(item for item in directories if not item.startswith("."))
        current_path = Path(current)
        paths = [current_path / filename for filename in filenames]
        fasta_files = [
            path
            for path in paths
            if re.search(r"\.(?:fna|fa|fasta)(?:\.gz)?$", path.name, re.IGNORECASE)
        ]
        annotation_files = [
            path
            for path in paths
            if re.search(r"\.(?:gff|gff3|gtf)(?:\.gz)?$", path.name, re.IGNORECASE)
        ]
        if not fasta_files or not annotation_files:
            continue
        report_files = [
            path
            for path in paths
            if "assembly_report" in path.name.casefold()
            or "sequence_report" in path.name.casefold()
            or path.name.casefold() in {"assembly_data_report.jsonl", "dataset_catalog.json"}
        ]
        metadata_candidates = list(report_files)
        metadata_candidates.extend(
            path
            for path in paths
            if path.suffix.casefold() in {".txt", ".json", ".jsonl"}
            and path.stat().st_size <= 5_000_000
        )
        metadata_text = _small_metadata_text(dict.fromkeys(metadata_candidates))
        candidate_dirs.append((current_path, fasta_files, annotation_files, report_files, metadata_text))

    if not candidate_dirs:
        raise BuildError(
            f"No directory under {genomes_root} contains both a genomic FASTA and GFF/GTF annotation"
        )

    resolved: dict[str, SpeciesInputs] = {}
    used_directories: set[Path] = set()
    for spec in SPECIES:
        matches: list[tuple[int, Path, list[Path], list[Path], list[Path], tuple[str, ...]]] = []
        for directory, fastas, annotations, reports, metadata_text in candidate_dirs:
            rank, evidence = species_match_evidence(spec, directory, metadata_text)
            if rank:
                matches.append((rank, directory, fastas, annotations, reports, evidence))
        if not matches:
            raise BuildError(
                f"Required species {spec.scientific_name} was not found under {genomes_root}. "
                f"Expected its scientific name, taxonomy ID {spec.taxid}, or a recognized directory alias."
            )
        best_rank = max(match[0] for match in matches)
        best = [match for match in matches if match[0] == best_rank]
        if len(best) != 1:
            choices = "\n  - ".join(f"{match[1]} ({'; '.join(match[5])})" for match in best)
            raise BuildError(f"Required species {spec.scientific_name} resolves ambiguously:\n  - {choices}")
        _, directory, fastas, annotations, reports, evidence = best[0]
        if directory in used_directories:
            raise BuildError(f"Species resolution reused directory unexpectedly: {directory}")
        used_directories.add(directory)
        resolved[spec.slug] = SpeciesInputs(
            spec=spec,
            directory=directory,
            fasta=choose_input_file(fastas, "genomic FASTA", directory),
            annotation=choose_input_file(annotations, "annotation", directory),
            reports=tuple(sorted(reports)),
            resolution_evidence=evidence,
        )
    return resolved


class IndexedFasta:
    """Small adapter around pyfaidx with conservative seqid resolution."""

    def __init__(self, path: Path, index_path: Path):
        try:
            from pyfaidx import Fasta  # type: ignore
        except ImportError as error:
            raise BuildError(
                "Indexed FASTA access requires pyfaidx. Install the project environment "
                "or run `python -m pip install pyfaidx`."
            ) from error
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fasta = Fasta(
                str(path),
                indexname=str(index_path),
                as_raw=True,
                sequence_always_upper=True,
                rebuild=True,
            )
        except Exception as error:
            compressed_hint = (
                " Standard gzip files must be converted to BGZF (or decompressed) for random access."
                if path.name.casefold().endswith(".gz")
                else ""
            )
            raise BuildError(f"Could not open/index FASTA {path} with pyfaidx: {error}.{compressed_hint}") from error
        self._keys = tuple(self._fasta.keys())
        self._aliases: dict[str, set[str]] = defaultdict(set)
        for key in self._keys:
            for alias in self._key_aliases(key):
                self._aliases[alias].add(key)

    @staticmethod
    def _key_aliases(value: str) -> set[str]:
        value = value.strip()
        aliases = {value, value.casefold()}
        if re.match(r"^[A-Z]{1,4}_?\d+\.\d+$", value, re.IGNORECASE):
            aliases.add(value.rsplit(".", 1)[0])
            aliases.add(value.rsplit(".", 1)[0].casefold())
        return aliases

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def resolve(self, seqid: str) -> str | None:
        if seqid in self._fasta:
            return seqid
        matches: set[str] = set()
        for alias in self._key_aliases(seqid):
            matches.update(self._aliases.get(alias, set()))
        return next(iter(matches)) if len(matches) == 1 else None

    def length(self, seqid: str) -> int:
        resolved = self.resolve(seqid)
        if resolved is None:
            raise KeyError(seqid)
        return len(self._fasta[resolved])

    def fetch(self, seqid: str, start: int, end: int) -> str:
        resolved = self.resolve(seqid)
        if resolved is None:
            raise KeyError(seqid)
        return str(self._fasta[resolved][start:end]).upper()

    def description(self, seqid: str) -> str:
        resolved = self.resolve(seqid)
        if resolved is None:
            return ""
        record = self._fasta[resolved]
        return str(getattr(record, "long_name", resolved))

    def close(self) -> None:
        close = getattr(self._fasta, "close", None)
        if close:
            close()


def classify_text(text: str, source: str, role: str = "", molecule: str = "") -> SequenceClassification:
    normalized = normalized_text(" ".join((text, role, molecule)))
    if re.search(r"(?:^|_)(?:chloroplast|plastid|plastome|chloroplastic)(?:_|$)", normalized):
        return SequenceClassification("chloroplast", source, role, molecule, False)
    if re.search(r"(?:^|_)(?:mitochondrion|mitochondrial|mitogenome)(?:_|$)", normalized):
        return SequenceClassification("mitochondrial", source, role, molecule, False)

    role_norm = normalized_text(role)
    molecule_norm = normalized_text(molecule)
    text_norm = normalized_text(text)
    nuclear_molecule = (
        molecule_norm in {"chromosome", "nuclear", "nuclear_chromosome"}
        or bool(re.search(r"(?:^|_)chromosome(?:_|$)", molecule_norm))
    )
    explicit_nuclear_text = (
        bool(re.search(r"(?:^|_)nuclear(?:_|$)", text_norm))
        or bool(re.search(r"(?:^|_)chromosome_(?:\d+|[a-z])(?:_|$)", text_norm))
    )
    if nuclear_molecule or explicit_nuclear_text:
        primary = role_norm in {"assembled_molecule", "chromosome"} or (
            not role_norm and bool(re.search(r"(?:^|_)chromosome_(?:\d+|[a-z])(?:_|$)", text_norm))
        )
        return SequenceClassification("nuclear", source, role, molecule, primary)
    return SequenceClassification("unknown", source, role, molecule, False)


def _report_row_classification(row: Mapping[str, object], source: str) -> SequenceClassification:
    folded = {normalized_text(str(key)): str(value) for key, value in row.items() if value is not None}

    def value(*keys: str) -> str:
        for key in keys:
            found = folded.get(normalized_text(key), "")
            if found and found.casefold() not in {"na", "none", "null"}:
                return found
        return ""

    role = value("sequence-role", "sequence_role", "role")
    molecule_type = value(
        "assigned-molecule-location/type",
        "assigned_molecule_location_type",
        "molecule_type",
        "assigned_molecule_type",
    )
    molecule = value("assigned-molecule", "assigned_molecule", "chr_name", "chromosome")
    combined = " ".join(str(item) for item in row.values())
    classification = classify_text(combined, source, role, molecule_type or molecule)
    if classification.compartment == "unknown" and role.casefold() == "unlocalized-scaffold" and molecule:
        classification = SequenceClassification("nuclear", source, role, molecule, False)
    return classification


def _report_aliases(row: Mapping[str, object]) -> set[str]:
    aliases: set[str] = set()
    key_words = (
        "sequence_name",
        "genbank_accession",
        "refseq_accession",
        "ucsc_style_name",
        "chr_name",
        "accession",
    )
    for key, raw_value in row.items():
        key_norm = normalized_text(str(key))
        if not any(word in key_norm for word in key_words):
            continue
        if isinstance(raw_value, list):
            values = raw_value
        else:
            values = [raw_value]
        for value in values:
            text_value = str(value).strip()
            if text_value and text_value.casefold() not in {"na", "none", "null"}:
                aliases.add(text_value)
    return aliases


def parse_assembly_report(path: Path) -> list[tuple[set[str], SequenceClassification]]:
    rows: list[tuple[set[str], SequenceClassification]] = []
    header: list[str] | None = None
    with open_text(path) as handle:
        for line in handle:
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("# ").split("\t")
                if "Sequence-Name" in candidate and "Sequence-Role" in candidate:
                    header = candidate
                continue
            columns = stripped.split("\t")
            if header is None:
                if len(columns) >= 9:
                    header = [
                        "Sequence-Name",
                        "Sequence-Role",
                        "Assigned-Molecule",
                        "Assigned-Molecule-Location/Type",
                        "GenBank-Accn",
                        "Relationship",
                        "RefSeq-Accn",
                        "Assembly-Unit",
                        "Sequence-Length",
                        "UCSC-style-name",
                    ][: len(columns)]
                else:
                    continue
            row = dict(zip(header, columns))
            aliases = {
                row.get(key, "").strip()
                for key in ("Sequence-Name", "GenBank-Accn", "RefSeq-Accn", "UCSC-style-name")
                if row.get(key, "").strip() not in {"", "na"}
            }
            if aliases:
                rows.append((aliases, _report_row_classification(row, f"assembly_report:{path.name}")))
    return rows


def _flatten_json(value: object, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(_flatten_json(child, child_prefix))
    elif isinstance(value, list):
        flattened[prefix] = value
    else:
        flattened[prefix] = value
    return flattened


def parse_sequence_report(path: Path) -> list[tuple[set[str], SequenceClassification]]:
    rows: list[tuple[set[str], SequenceClassification]] = []
    with open_text(path) as handle:
        first = handle.readline()
        if not first:
            return rows
        stripped = first.lstrip()
        if stripped.startswith("{"):
            lines = [first]
            lines.extend(handle)
            for line in lines:
                try:
                    row = _flatten_json(json.loads(line))
                except json.JSONDecodeError:
                    continue
                aliases = _report_aliases(row)
                if aliases:
                    rows.append((aliases, _report_row_classification(row, f"sequence_report:{path.name}")))
            return rows
        delimiter = "\t" if "\t" in first else ","
        reader = csv.DictReader([first, *handle], delimiter=delimiter)
        for raw_row in reader:
            row = {str(key): value for key, value in raw_row.items()}
            aliases = _report_aliases(row)
            if aliases:
                rows.append((aliases, _report_row_classification(row, f"sequence_report:{path.name}")))
    return rows


def report_classifications(reports: Sequence[Path]) -> dict[str, SequenceClassification]:
    classifications: dict[str, SequenceClassification] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        name = report.name.casefold()
        try:
            parsed = parse_assembly_report(report) if "assembly_report" in name else parse_sequence_report(report)
        except (OSError, csv.Error, UnicodeError) as error:
            raise BuildError(f"Could not parse sequence metadata report {report}: {error}") from error
        for aliases, classification in parsed:
            for alias in aliases:
                for key in IndexedFasta._key_aliases(alias):
                    existing = classifications.get(key)
                    if existing and existing.compartment != classification.compartment:
                        conflicts[key].update((existing.compartment, classification.compartment))
                    else:
                        classifications[key] = classification
    if conflicts:
        details = "; ".join(f"{key}={sorted(values)}" for key, values in sorted(conflicts.items()))
        raise BuildError(f"Conflicting compartment assignments in NCBI reports: {details}")
    return classifications


def annotation_hint(feature: Feature) -> str:
    values = [feature.feature_type]
    for key in ("genome", "organelle", "chromosome", "Name", "note", "mol_type", "gbkey"):
        values.extend(f"{key} {value}" for value in attribute_values(feature, key))
    return " ".join(values)


def add_merged_interval(target: list[Interval], start: int, end: int) -> None:
    if end <= start:
        return
    if target and start >= target[-1].start and start <= target[-1].end:
        previous = target[-1]
        target[-1] = Interval(previous.start, max(previous.end, end))
    else:
        target.append(Interval(start, end))


def merge_intervals(intervals: Sequence[Interval], sequence_length: int | None = None) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for interval in ordered:
        start = max(0, interval.start)
        end = min(sequence_length, interval.end) if sequence_length is not None else interval.end
        if end <= start:
            continue
        if merged and start <= merged[-1].end:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, end))
        else:
            merged.append(Interval(start, end))
    return merged


def complement_intervals(mask: Sequence[Interval], sequence_length: int) -> list[Interval]:
    eligible: list[Interval] = []
    cursor = 0
    for interval in merge_intervals(mask, sequence_length):
        if cursor < interval.start:
            eligible.append(Interval(cursor, interval.start))
        cursor = max(cursor, interval.end)
    if cursor < sequence_length:
        eligible.append(Interval(cursor, sequence_length))
    return eligible


def load_annotation(
    path: Path, gene_buffer: int
) -> tuple[list[Feature], dict[str, list[Interval]], list[Feature], dict[str, str]]:
    pseudogenes: list[Feature] = []
    masks: dict[str, list[Interval]] = defaultdict(list)
    genes: list[Feature] = []
    hints: dict[str, list[str]] = defaultdict(list)
    for feature in iter_gff(path):
        is_pseudo, explicit = pseudogene_rule(feature)
        if is_pseudo:
            feature.explicit_pseudogene = explicit
            pseudogenes.append(feature)
        if is_mask_feature(feature):
            add_merged_interval(masks[feature.seqid], feature.start - gene_buffer, feature.end + gene_buffer)
        if is_nearest_gene_feature(feature):
            genes.append(feature)
        if feature.feature_type.casefold() in {"region", "chromosome", "sequence"}:
            hints[feature.seqid].append(annotation_hint(feature))
    merged_masks = {seqid: merge_intervals(intervals) for seqid, intervals in masks.items()}
    hint_text = {seqid: " ".join(values) for seqid, values in hints.items()}
    return deduplicate_pseudogenes(pseudogenes), merged_masks, genes, hint_text


def lookup_report_classification(
    seqid: str, reports: Mapping[str, SequenceClassification]
) -> SequenceClassification | None:
    matches = {reports[alias] for alias in IndexedFasta._key_aliases(seqid) if alias in reports}
    if len(matches) > 1:
        compartments = {match.compartment for match in matches}
        if len(compartments) > 1:
            raise BuildError(f"Ambiguous report classification for reference sequence {seqid}: {compartments}")
    return next(iter(matches)) if matches else None


def classify_reference_sequences(
    fasta: object,
    reports: Mapping[str, SequenceClassification],
    gff_hints: Mapping[str, str],
) -> dict[str, SequenceClassification]:
    """Apply report > FASTA description > GFF metadata precedence."""

    result: dict[str, SequenceClassification] = {}
    for seqid in fasta.keys:  # type: ignore[attr-defined]
        report = lookup_report_classification(seqid, reports)
        if report is not None:
            result[seqid] = report
            continue
        description = fasta.description(seqid)  # type: ignore[attr-defined]
        from_fasta = classify_text(description, "FASTA_description")
        if from_fasta.compartment != "unknown":
            result[seqid] = from_fasta
            continue
        hint_matches = {
            value
            for hint_seqid, value in gff_hints.items()
            if set(IndexedFasta._key_aliases(seqid)) & set(IndexedFasta._key_aliases(hint_seqid))
        }
        hint = next(iter(hint_matches)) if len(hint_matches) == 1 else ""
        from_gff = classify_text(hint, "GFF_metadata")
        result[seqid] = from_gff
    return result


def classification_for_seqid(
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    seqid: str,
) -> SequenceClassification:
    resolved = fasta.resolve(seqid)  # type: ignore[attr-defined]
    if resolved is None:
        return SequenceClassification("unknown", "annotation seqid absent from FASTA")
    return classifications.get(resolved, SequenceClassification("unknown", "no classification evidence"))


def reverse_complement(sequence: str) -> str:
    translation = str.maketrans(
        "ACGTRYMKBDHVNUWSacgtrymkbdhvnuws",
        "TGCAYRKMVHDBNAWStgcayrkmvhdbnaws",
    )
    return sequence.translate(translation)[::-1]


def sequence_fractions(sequence: str) -> tuple[float, float]:
    if not sequence:
        return 0.0, 0.0
    upper = sequence.upper()
    length = len(upper)
    return (upper.count("G") + upper.count("C")) / length, upper.count("N") / length


def pseudogene_window_bounds(feature: Feature) -> tuple[int, int, int]:
    """Return requested [start, end) and the 1-based genomic anchor."""

    if feature.strand == "+":
        return feature.start - UPSTREAM_LENGTH, feature.start + DOWNSTREAM_LENGTH, feature.start + 1
    if feature.strand == "-":
        # end is already the one-based inclusive GFF end expressed as a
        # zero-based exclusive coordinate. The anchor base is end - 1.
        return feature.end - DOWNSTREAM_LENGTH, feature.end + UPSTREAM_LENGTH, feature.end
    raise ValueError(f"Pseudogene has unsupported strand {feature.strand!r}")


def extract_requested_window(
    fasta: object,
    seqid: str,
    requested_start: int,
    requested_end: int,
    pad_boundaries: bool,
) -> tuple[str, int, int, int, int]:
    """Fetch a window, returning sequence, clipped bounds, and N padding."""

    sequence_length = fasta.length(seqid)  # type: ignore[attr-defined]
    left_pad = max(0, -requested_start)
    right_pad = max(0, requested_end - sequence_length)
    if (left_pad or right_pad) and not pad_boundaries:
        raise ValueError("boundary_truncated")
    clipped_start = max(0, requested_start)
    clipped_end = min(sequence_length, requested_end)
    sequence = "N" * left_pad
    sequence += fasta.fetch(seqid, clipped_start, clipped_end)  # type: ignore[attr-defined]
    sequence += "N" * right_pad
    if len(sequence) != requested_end - requested_start:
        raise ValueError(
            f"reference_fetch_length_mismatch: expected {requested_end - requested_start}, got {len(sequence)}"
        )
    return sequence, clipped_start, clipped_end, left_pad, right_pad


def prepare_pseudogene_candidates(
    features: Sequence[Feature],
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    pad_boundaries: bool,
) -> list[CandidateStatus]:
    statuses: list[CandidateStatus] = []
    for feature in features:
        resolved = fasta.resolve(feature.seqid)  # type: ignore[attr-defined]
        classification = classification_for_seqid(fasta, classifications, feature.seqid)
        if resolved is None:
            statuses.append(
                CandidateStatus(
                    feature,
                    "unknown",
                    classification.evidence,
                    False,
                    "missing_reference_sequence",
                )
            )
            continue
        try:
            requested_start, requested_end, _ = pseudogene_window_bounds(feature)
        except ValueError:
            statuses.append(
                CandidateStatus(
                    feature,
                    classification.compartment,
                    classification.evidence,
                    False,
                    "invalid_strand",
                )
            )
            continue
        sequence_length = fasta.length(resolved)  # type: ignore[attr-defined]
        boundary = requested_start < 0 or requested_end > sequence_length
        statuses.append(
            CandidateStatus(
                feature,
                classification.compartment,
                classification.evidence,
                not boundary or pad_boundaries,
                "" if not boundary or pad_boundaries else "boundary_truncated",
                requested_start,
                requested_end,
            )
        )
    return statuses


def deterministic_cap(items: Sequence[T], cap: int, seed: int, *parts: str) -> list[T]:
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    ordered = list(items)
    if len(ordered) <= cap:
        return ordered
    rng = random.Random(stable_seed(seed, *parts))
    selected_indices = sorted(rng.sample(range(len(ordered)), cap))
    return [ordered[index] for index in selected_indices]


def compartment_is_selected(compartment: str, include_organelles: bool, include_unknown: bool) -> bool:
    if compartment == "nuclear":
        return True
    if compartment in {"chloroplast", "mitochondrial"}:
        return include_organelles
    return include_unknown


def make_dataset_id(spec: SpeciesSpec, category: str, identity: str) -> str:
    return f"{spec.slug}__{category}__{safe_id(identity)}"


def pseudogene_record(
    spec: SpeciesSpec,
    status: CandidateStatus,
    fasta: object,
    pad_boundaries: bool,
) -> WindowRecord:
    feature = status.feature
    assert status.requested_start is not None and status.requested_end is not None
    sequence, clipped_start, clipped_end, left_pad, right_pad = extract_requested_window(
        fasta,
        feature.seqid,
        status.requested_start,
        status.requested_end,
        pad_boundaries,
    )
    _, _, anchor_1based = pseudogene_window_bounds(feature)
    if feature.strand == "-":
        sequence = reverse_complement(sequence)
    if len(sequence) != WINDOW_LENGTH:
        raise BuildError(
            f"Pseudogene {canonical_feature_id(feature)} produced {len(sequence)} bp instead of {WINDOW_LENGTH}"
        )
    gc_fraction, n_fraction = sequence_fractions(sequence)
    gene_id = canonical_feature_id(feature)
    ncbi_gene_id = next(
        (value for value, evidence in stable_identifiers(feature) if evidence == "NCBI_GeneID"), ""
    )
    subtype = first_attribute(feature, "pseudogene", "pseudogene_type", "pseudogene_subtype")
    return WindowRecord(
        dataset_id=make_dataset_id(
            spec,
            "pseudogene",
            f"{gene_id}_{feature.seqid}_{feature.start + 1}_{feature.end}_{feature.strand}",
        ),
        species=spec.scientific_name,
        locus_category="annotated_pseudogene",
        seqid=feature.seqid,
        start=clipped_start,
        end=clipped_end,
        strand=feature.strand,
        compartment=status.compartment,
        anchor_type="annotated_5prime_boundary",
        anchor_genomic_1based=anchor_1based,
        anchor_offset_0based=ANCHOR_OFFSET,
        gc_fraction=gc_fraction,
        n_fraction=n_fraction,
        sequence=sequence,
        extra={
            "requested_start_0based": status.requested_start,
            "requested_end_0based": status.requested_end,
            "left_padding_bp": left_pad,
            "right_padding_bp": right_pad,
            "locus_start_1based": feature.start + 1,
            "locus_end_1based": feature.end,
            "stable_id": first_attribute(feature, "ID", "gene_id"),
            "gene_name": first_attribute(feature, "Name", "gene", "gene_name"),
            "locus_tag": first_attribute(feature, "locus_tag"),
            "ncbi_gene_id": ncbi_gene_id,
            "gene_biotype": first_attribute(feature, "gene_biotype", "gene_type", "biotype"),
            "pseudogene_subtype": subtype,
            "note": first_attribute(feature, "Note", "note", "description"),
            "annotation_feature_type": feature.feature_type,
            "annotation_source": feature.source,
            "annotation_line": feature.line_number,
            "classification_evidence": status.classification_evidence,
            "original_attributes": feature.attributes_raw,
            "exclusion_status": "emitted",
        },
    )


def _fasta_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise BuildError(f"PGB directory is not accessible: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and re.search(r"\.(?:fa|fasta|fna)(?:\.gz)?$", path.name, re.IGNORECASE)
    )


def path_matches_species(path: Path, spec: SpeciesSpec, root: Path) -> bool:
    relative = normalized_text(str(path.relative_to(root)))
    names = (normalized_text(spec.scientific_name), spec.slug, *map(normalized_text, spec.aliases))
    return any(
        len(name) >= 4 and re.search(rf"(?:^|_){re.escape(name)}(?:_|$)", relative)
        for name in names
    )


def discover_pgb_fastas(pgb_dir: Path) -> dict[str, list[Path]]:
    files = _fasta_paths(pgb_dir)
    if not files:
        raise BuildError(f"No PGB FASTA files were found under {pgb_dir}")
    result: dict[str, list[Path]] = {}
    for spec in SPECIES:
        matches = [path for path in files if path_matches_species(path, spec, pgb_dir)]
        if not matches:
            raise BuildError(
                f"No PGB FASTA path could be associated with {spec.scientific_name} under {pgb_dir}. "
                "Include the species scientific name or a conventional species alias in its path."
            )
        result[spec.slug] = matches
    return result


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            elif header is not None:
                chunks.append("".join(line.split()))
    if header is not None:
        yield header, "".join(chunks).upper()


def load_pgb_records(paths: Sequence[Path]) -> tuple[list[PgbRecord], list[str]]:
    records: dict[str, PgbRecord] = {}
    warnings: list[str] = []
    for path in paths:
        for header, sequence in iter_fasta(path):
            gene_id = header.split("|", 1)[0].strip()
            if not gene_id:
                warnings.append(f"empty PGB identifier in {path}")
                continue
            existing = records.get(gene_id)
            if existing is None:
                records[gene_id] = PgbRecord(gene_id, sequence, {str(path)})
            else:
                existing.source_files.add(str(path))
                if existing.sequence != sequence:
                    existing.conflicting_sequences = True
                    warnings.append(f"PGB identifier {gene_id} has conflicting sequences across files")
    return [records[key] for key in sorted(records)], warnings


def normalized_identifier_forms(identifier: str) -> set[str]:
    raw = unquote(identifier).strip()
    forms = {raw.casefold()}
    for prefix in ("gene:", "gene-", "locus:"):
        if raw.casefold().startswith(prefix):
            forms.add(raw[len(prefix) :].casefold())
    return {form for form in forms if form}


def build_annotation_identifier_index(
    genes: Sequence[Feature],
) -> tuple[dict[str, set[int]], dict[tuple[int, str], str]]:
    index: dict[str, set[int]] = defaultdict(set)
    evidence_by_match: dict[tuple[int, str], str] = {}
    for feature_index, feature in enumerate(genes):
        for identifier, evidence in stable_identifiers(feature):
            for normalized in normalized_identifier_forms(identifier):
                index[normalized].add(feature_index)
                evidence_by_match[(feature_index, normalized)] = evidence
    return index, evidence_by_match


def extract_gene_anchor_window(fasta: object, feature: Feature) -> str | None:
    try:
        requested_start, requested_end, _ = pseudogene_window_bounds(feature)
        sequence, _, _, _, _ = extract_requested_window(
            fasta, feature.seqid, requested_start, requested_end, False
        )
    except (KeyError, ValueError):
        return None
    return reverse_complement(sequence) if feature.strand == "-" else sequence


def build_exact_sequence_index(fasta: object, genes: Sequence[Feature]) -> dict[str, set[int]]:
    sequence_index: dict[str, set[int]] = defaultdict(set)
    for index, feature in enumerate(genes):
        sequence = extract_gene_anchor_window(fasta, feature)
        if sequence is not None and len(sequence) == WINDOW_LENGTH:
            sequence_index[hashlib.sha256(sequence.upper().encode("ascii", "replace")).hexdigest()].add(index)
    return sequence_index


def audit_pgb(
    records: Sequence[PgbRecord],
    genes: Sequence[Feature],
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    exact_sequence_mapping: bool,
) -> tuple[list[PgbMapping], list[tuple[float, float]]]:
    identifier_index, evidence_index = build_annotation_identifier_index(genes)
    sequence_index = build_exact_sequence_index(fasta, genes) if exact_sequence_mapping else {}
    mappings: list[PgbMapping] = []
    metrics: list[tuple[float, float]] = []
    for record in records:
        metrics.append(sequence_fractions(record.sequence))
        candidate_indices: set[int] = set()
        matched_forms: list[str] = []
        for form in normalized_identifier_forms(record.gene_id):
            found = identifier_index.get(form, set())
            if found:
                candidate_indices.update(found)
                matched_forms.append(form)
        evidence = ""
        if len(candidate_indices) == 1:
            feature_index = next(iter(candidate_indices))
            evidence_types = {
                evidence_index.get((feature_index, form), "stable_identifier") for form in matched_forms
            }
            evidence = "identifier:" + ",".join(sorted(evidence_types))
        elif not candidate_indices and exact_sequence_mapping and record.sequence:
            digest = hashlib.sha256(record.sequence.upper().encode("ascii", "replace")).hexdigest()
            candidate_indices = set(sequence_index.get(digest, set()))
            if len(candidate_indices) == 1:
                feature_index = next(iter(candidate_indices))
                evidence = "exact_oriented_6000bp_sequence"
        if len(candidate_indices) == 1:
            feature_index = next(iter(candidate_indices))
            feature = genes[feature_index]
            classification = classification_for_seqid(fasta, classifications, feature.seqid)
            mappings.append(
                PgbMapping(
                    record.gene_id,
                    True,
                    classification.compartment,
                    evidence,
                    canonical_feature_id(feature),
                    feature.seqid,
                    "conflicting PGB sequences" if record.conflicting_sequences else "",
                )
            )
        else:
            note = "ambiguous identifier mapping" if candidate_indices else "no exact annotation mapping"
            if record.conflicting_sequences:
                note += "; conflicting PGB sequences"
            mappings.append(PgbMapping(record.gene_id, False, "unmapped", "", note=note))
    return mappings, metrics


@dataclass
class IntergenicCandidate:
    seqid: str
    start: int
    end: int
    sequence: str
    gc_fraction: float
    n_fraction: float
    compartment: str


class NearestGeneIndex:
    def __init__(
        self,
        features: Sequence[Feature],
        seqid_resolver: Callable[[str], str | None] | None = None,
    ):
        self._by_start: dict[str, list[Feature]] = defaultdict(list)
        self._by_end: dict[str, list[Feature]] = defaultdict(list)
        for feature in features:
            seqid = seqid_resolver(feature.seqid) if seqid_resolver else feature.seqid
            if seqid is None:
                continue
            self._by_start[seqid].append(feature)
            self._by_end[seqid].append(feature)
        self._starts: dict[str, list[int]] = {}
        self._ends: dict[str, list[int]] = {}
        for seqid, items in self._by_start.items():
            items.sort(key=lambda feature: (feature.start, feature.end))
            self._starts[seqid] = [feature.start for feature in items]
        for seqid, items in self._by_end.items():
            items.sort(key=lambda feature: (feature.end, feature.start))
            self._ends[seqid] = [feature.end for feature in items]

    def nearest(self, seqid: str, start: int, end: int) -> tuple[Feature | None, int | None]:
        candidates: list[tuple[int, Feature]] = []
        by_end = self._by_end.get(seqid, [])
        end_values = self._ends.get(seqid, [])
        previous_index = bisect.bisect_right(end_values, start) - 1
        if previous_index >= 0:
            feature = by_end[previous_index]
            candidates.append((max(0, start - feature.end), feature))
        by_start = self._by_start.get(seqid, [])
        start_values = self._starts.get(seqid, [])
        next_index = bisect.bisect_left(start_values, end)
        if next_index < len(by_start):
            feature = by_start[next_index]
            candidates.append((max(0, feature.start - end), feature))
        # Defensive overlap check for callers that use an unbuffered index.
        overlap_index = bisect.bisect_right(start_values, start) - 1
        if overlap_index >= 0 and by_start[overlap_index].end > start:
            candidates.append((0, by_start[overlap_index]))
        if not candidates:
            return None, None
        distance, feature = min(
            candidates,
            key=lambda item: (item[0], canonical_feature_id(item[1]), item[1].start),
        )
        return feature, distance


def map_masks_to_reference(
    fasta: object,
    masks: Mapping[str, Sequence[Interval]],
) -> tuple[dict[str, list[Interval]], list[str]]:
    mapped: dict[str, list[Interval]] = defaultdict(list)
    missing: list[str] = []
    for seqid, intervals in masks.items():
        resolved = fasta.resolve(seqid)  # type: ignore[attr-defined]
        if resolved is None:
            missing.append(seqid)
            continue
        mapped[resolved].extend(intervals)
    return {
        seqid: merge_intervals(intervals, fasta.length(seqid))  # type: ignore[attr-defined]
        for seqid, intervals in mapped.items()
    }, sorted(set(missing))


def calculate_intergenic_availability(
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    masks: Mapping[str, Sequence[Interval]],
    include_organelles: bool,
    include_unknown: bool,
) -> tuple[dict[str, list[Interval]], int, int, int]:
    allowed = {"nuclear"}
    if include_organelles:
        allowed.update(("chloroplast", "mitochondrial"))
    if include_unknown:
        allowed.add("unknown")
    return calculate_availability_for_compartments(fasta, classifications, masks, allowed)


def calculate_availability_for_compartments(
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    masks: Mapping[str, Sequence[Interval]],
    allowed_compartments: set[str],
) -> tuple[dict[str, list[Interval]], int, int, int]:
    mapped_masks, _ = map_masks_to_reference(fasta, masks)
    eligible: dict[str, list[Interval]] = {}
    total_nt = 0
    capacity = 0
    stranded_nt = 0
    for seqid in sorted(fasta.keys):  # type: ignore[attr-defined]
        classification = classifications.get(
            seqid, SequenceClassification("unknown", "no classification evidence")
        )
        if classification.compartment not in allowed_compartments:
            continue
        intervals = complement_intervals(mapped_masks.get(seqid, []), fasta.length(seqid))  # type: ignore[attr-defined]
        eligible[seqid] = intervals
        total_nt += sum(interval.length for interval in intervals)
        capacity += sum(interval.capacity for interval in intervals)
        stranded_nt += sum(interval.length % WINDOW_LENGTH for interval in intervals)
    return eligible, total_nt, capacity, stranded_nt


def interval_slot(intervals: Sequence[Interval], ordinal: int) -> tuple[int, int]:
    if ordinal < 0:
        raise IndexError(ordinal)
    cursor = ordinal
    for interval in intervals:
        if cursor < interval.capacity:
            start = interval.start + cursor * WINDOW_LENGTH
            return start, start + WINDOW_LENGTH
        cursor -= interval.capacity
    raise IndexError(ordinal)


def balanced_quotas(
    total: int,
    capacities: Mapping[str, int],
    primary_sequences: set[str] | None = None,
) -> dict[str, int]:
    quotas = {seqid: 0 for seqid in capacities}
    remaining = min(total, sum(capacities.values()))
    if remaining <= 0:
        return quotas
    primary_sequences = primary_sequences or set()
    primary = [seqid for seqid in sorted(capacities) if seqid in primary_sequences and capacities[seqid] > 0]
    secondary = [seqid for seqid in sorted(capacities) if seqid not in primary_sequences and capacities[seqid] > 0]

    def allocate(pool: list[str], amount: int) -> int:
        active = list(pool)
        left = amount
        while left > 0 and active:
            progressed = False
            for seqid in list(active):
                if left == 0:
                    break
                if quotas[seqid] < capacities[seqid]:
                    quotas[seqid] += 1
                    left -= 1
                    progressed = True
                else:
                    active.remove(seqid)
            if not progressed:
                break
        return left

    if primary:
        remaining = allocate(primary, remaining)
    if remaining:
        remaining = allocate(secondary, remaining)
    return quotas


def _n_bin(value: float) -> int:
    if value == 0:
        return 0
    if value < 0.001:
        return 1
    if value < 0.01:
        return 2
    if value < 0.05:
        return 3
    return 4


def match_candidate_metrics(
    candidates: Sequence[IntergenicCandidate],
    target_metrics: Sequence[tuple[float, float]],
    count: int,
    rng: random.Random,
) -> list[IntergenicCandidate]:
    if count >= len(candidates):
        return list(candidates)
    if not target_metrics:
        indices = rng.sample(range(len(candidates)), count)
        return [candidates[index] for index in indices]

    buckets: dict[tuple[int, int], list[IntergenicCandidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[(min(50, int(candidate.gc_fraction * 50)), _n_bin(candidate.n_fraction))].append(candidate)
    for values in buckets.values():
        rng.shuffle(values)

    ordered_targets = sorted(target_metrics, key=lambda value: (value[0], value[1]))
    if len(ordered_targets) > count:
        target_indices = [min(len(ordered_targets) - 1, (index * len(ordered_targets)) // count) for index in range(count)]
        desired = [ordered_targets[index] for index in target_indices]
    else:
        desired = [ordered_targets[index % len(ordered_targets)] for index in range(count)]

    selected: list[IntergenicCandidate] = []
    for gc_target, n_target in desired:
        target_key = (min(50, int(gc_target * 50)), _n_bin(n_target))
        available_keys = [key for key, values in buckets.items() if values]
        if not available_keys:
            break
        key = min(
            available_keys,
            key=lambda value: (
                abs(value[0] - target_key[0]) + 3 * abs(value[1] - target_key[1]),
                value,
            ),
        )
        selected.append(buckets[key].pop())
    return selected


def sample_intergenic_candidates(
    spec: SpeciesSpec,
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    intervals: Mapping[str, Sequence[Interval]],
    maximum: int,
    seed: int,
    pgb_metrics: Sequence[tuple[float, float]],
) -> list[IntergenicCandidate]:
    capacities = {
        seqid: sum(interval.capacity for interval in seq_intervals)
        for seqid, seq_intervals in intervals.items()
    }
    primary = {
        seqid
        for seqid in capacities
        if classifications.get(seqid, SequenceClassification("unknown", "")).primary_chromosome
    }
    quotas = balanced_quotas(maximum, capacities, primary)
    selected: list[IntergenicCandidate] = []
    sorted_targets = sorted(pgb_metrics, key=lambda value: (value[0], value[1]))
    target_cursor = 0
    for seqid in sorted(quotas):
        quota = quotas[seqid]
        if quota == 0:
            continue
        capacity = capacities[seqid]
        rng = random.Random(stable_seed(seed, spec.slug, "intergenic", seqid))
        pool_size = min(capacity, max(quota, quota * 4, quota + 20))
        ordinals = rng.sample(range(capacity), pool_size) if pool_size < capacity else list(range(capacity))
        pool: list[IntergenicCandidate] = []
        for ordinal in ordinals:
            start, end = interval_slot(intervals[seqid], ordinal)
            sequence = fasta.fetch(seqid, start, end)  # type: ignore[attr-defined]
            if len(sequence) != WINDOW_LENGTH:
                raise BuildError(
                    f"Intergenic window {seqid}:{start}-{end} returned {len(sequence)} bp"
                )
            gc_fraction, n_fraction = sequence_fractions(sequence)
            pool.append(
                IntergenicCandidate(
                    seqid,
                    start,
                    end,
                    sequence,
                    gc_fraction,
                    n_fraction,
                    classifications.get(
                        seqid, SequenceClassification("unknown", "no classification evidence")
                    ).compartment,
                )
            )
        if sorted_targets:
            targets = [
                sorted_targets[(target_cursor + index * max(1, len(sorted_targets) // quota)) % len(sorted_targets)]
                for index in range(quota)
            ]
            target_cursor += quota
        else:
            targets = []
        selected.extend(match_candidate_metrics(pool, targets, quota, rng))
    return sorted(selected, key=lambda item: (item.seqid, item.start, item.end))


def intergenic_records(
    spec: SpeciesSpec,
    candidates: Sequence[IntergenicCandidate],
    nearest_genes: NearestGeneIndex,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    for candidate in candidates:
        nearest, distance = nearest_genes.nearest(candidate.seqid, candidate.start, candidate.end)
        identity = f"{candidate.seqid}_{candidate.start + 1}_{candidate.end}"
        records.append(
            WindowRecord(
                dataset_id=make_dataset_id(spec, "intergenic", identity),
                species=spec.scientific_name,
                locus_category="intergenic_locus",
                seqid=candidate.seqid,
                start=candidate.start,
                end=candidate.end,
                strand="+",
                compartment=candidate.compartment,
                anchor_type="synthetic_intergenic",
                anchor_genomic_1based=candidate.start + ANCHOR_OFFSET + 1,
                anchor_offset_0based=ANCHOR_OFFSET,
                gc_fraction=candidate.gc_fraction,
                n_fraction=candidate.n_fraction,
                sequence=candidate.sequence,
                extra={
                    "nearest_gene_id": canonical_feature_id(nearest) if nearest else "",
                    "nearest_gene_feature_type": nearest.feature_type if nearest else "",
                    "nearest_gene_distance_bp": "" if distance is None else distance,
                    "exclusion_status": "emitted",
                },
            )
        )
    return records


BASE_METADATA_COLUMNS = [
    "species",
    "dataset_id",
    "locus_category",
    "seqid",
    "window_start_0based",
    "window_end_0based_exclusive",
    "window_start_1based",
    "window_end_1based_inclusive",
    "strand",
    "compartment",
    "anchor_type",
    "anchor_genomic_1based",
    "anchor_offset_0based",
    "sequence_length",
    "gc_fraction",
    "n_fraction",
    "exclusion_status",
]

PSEUDOGENE_EXTRA_COLUMNS = [
    "requested_start_0based",
    "requested_end_0based",
    "left_padding_bp",
    "right_padding_bp",
    "locus_start_1based",
    "locus_end_1based",
    "stable_id",
    "gene_name",
    "locus_tag",
    "ncbi_gene_id",
    "gene_biotype",
    "pseudogene_subtype",
    "note",
    "annotation_feature_type",
    "annotation_source",
    "annotation_line",
    "classification_evidence",
    "original_attributes",
]

INTERGENIC_EXTRA_COLUMNS = [
    "nearest_gene_id",
    "nearest_gene_feature_type",
    "nearest_gene_distance_bp",
]


def window_row(record: WindowRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "species": record.species,
        "dataset_id": record.dataset_id,
        "locus_category": record.locus_category,
        "seqid": record.seqid,
        "window_start_0based": record.start,
        "window_end_0based_exclusive": record.end,
        "window_start_1based": record.start + 1,
        "window_end_1based_inclusive": record.end,
        "strand": record.strand,
        "compartment": record.compartment,
        "anchor_type": record.anchor_type,
        "anchor_genomic_1based": record.anchor_genomic_1based,
        "anchor_offset_0based": record.anchor_offset_0based,
        "sequence_length": len(record.sequence),
        "gc_fraction": f"{record.gc_fraction:.6f}",
        "n_fraction": f"{record.n_fraction:.6f}",
        "exclusion_status": record.extra.get("exclusion_status", "emitted"),
    }
    row.update(record.extra)
    return row


def fasta_header(record: WindowRecord) -> str:
    return "|".join(
        (
            record.dataset_id,
            f"category={record.locus_category}",
            f"species={record.species.replace(' ', '_')}",
            f"seqid={record.seqid}",
            f"window={record.start + 1}-{record.end}",
            f"strand={record.strand}",
            f"compartment={record.compartment}",
            f"anchor_type={record.anchor_type}",
            f"anchor_offset={record.anchor_offset_0based}",
        )
    )


def write_fasta(path: Path, records: Sequence[WindowRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{fasta_header(record)}\n")
            sequence = record.sequence
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_emitted_records(
    pseudogenes: Sequence[WindowRecord], intergenic: Sequence[WindowRecord]
) -> list[str]:
    failures: list[str] = []
    all_records = [*pseudogenes, *intergenic]
    if len({record.dataset_id for record in all_records}) != len(all_records):
        failures.append("dataset IDs are not unique within the species output")
    for record in all_records:
        label = record.dataset_id
        if len(record.sequence) != WINDOW_LENGTH:
            failures.append(f"{label}: sequence length is {len(record.sequence)}, expected 6000")
        if record.anchor_offset_0based != ANCHOR_OFFSET:
            failures.append(f"{label}: anchor offset is not 5000")
        if record.start < 0 or record.end < record.start:
            failures.append(f"{label}: invalid clipped genomic coordinates {record.start}-{record.end}")

    for record in pseudogenes:
        requested_start = record.extra.get("requested_start_0based")
        requested_end = record.extra.get("requested_end_0based")
        if not isinstance(requested_start, int) or not isinstance(requested_end, int):
            failures.append(f"{record.dataset_id}: requested pseudogene coordinates are missing")
            continue
        if requested_end - requested_start != WINDOW_LENGTH:
            failures.append(f"{record.dataset_id}: requested pseudogene span is not 6000 bp")
        if record.strand == "+":
            expected_anchor = requested_start + ANCHOR_OFFSET + 1
        elif record.strand == "-":
            expected_anchor = requested_end - ANCHOR_OFFSET
        else:
            failures.append(f"{record.dataset_id}: invalid pseudogene strand {record.strand!r}")
            continue
        if record.anchor_genomic_1based != expected_anchor:
            failures.append(f"{record.dataset_id}: genomic anchor is inconsistent with strand/window")

    by_sequence: dict[str, list[WindowRecord]] = defaultdict(list)
    for record in intergenic:
        if record.end - record.start != WINDOW_LENGTH:
            failures.append(f"{record.dataset_id}: intergenic coordinate span is not 6000 bp")
        if record.anchor_type != "synthetic_intergenic":
            failures.append(f"{record.dataset_id}: intergenic anchor type is not synthetic_intergenic")
        if record.anchor_genomic_1based != record.start + ANCHOR_OFFSET + 1:
            failures.append(f"{record.dataset_id}: synthetic anchor is inconsistent with coordinates")
        by_sequence[record.seqid].append(record)
    for seqid, records in by_sequence.items():
        ordered = sorted(records, key=lambda record: (record.start, record.end))
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.start:
                failures.append(
                    f"intergenic windows overlap on {seqid}: {left.dataset_id}, {right.dataset_id}"
                )
    return failures


def pseudogene_audit_row(status: CandidateStatus, emitted: bool, reason: str = "") -> dict[str, object]:
    feature = status.feature
    requested_start = "" if status.requested_start is None else status.requested_start
    requested_end = "" if status.requested_end is None else status.requested_end
    ncbi_gene_id = next(
        (value for value, evidence in stable_identifiers(feature) if evidence == "NCBI_GeneID"), ""
    )
    return {
        "species": "",
        "dataset_id": "",
        "locus_category": "annotated_pseudogene",
        "seqid": feature.seqid,
        "window_start_0based": max(0, status.requested_start or 0),
        "window_end_0based_exclusive": max(0, status.requested_end or 0),
        "window_start_1based": max(0, status.requested_start or 0) + 1,
        "window_end_1based_inclusive": max(0, status.requested_end or 0),
        "strand": feature.strand,
        "compartment": status.compartment,
        "anchor_type": "annotated_5prime_boundary",
        "anchor_genomic_1based": feature.start + 1 if feature.strand == "+" else feature.end,
        "anchor_offset_0based": ANCHOR_OFFSET,
        "sequence_length": WINDOW_LENGTH if status.eligible else "",
        "gc_fraction": "",
        "n_fraction": "",
        "exclusion_status": "emitted" if emitted else (reason or status.exclusion_reason or "not_emitted"),
        "requested_start_0based": requested_start,
        "requested_end_0based": requested_end,
        "left_padding_bp": max(0, -(status.requested_start or 0)) if status.requested_start is not None else "",
        "right_padding_bp": "",
        "locus_start_1based": feature.start + 1,
        "locus_end_1based": feature.end,
        "stable_id": first_attribute(feature, "ID", "gene_id"),
        "gene_name": first_attribute(feature, "Name", "gene", "gene_name"),
        "locus_tag": first_attribute(feature, "locus_tag"),
        "ncbi_gene_id": ncbi_gene_id,
        "gene_biotype": first_attribute(feature, "gene_biotype", "gene_type", "biotype"),
        "pseudogene_subtype": first_attribute(
            feature, "pseudogene", "pseudogene_type", "pseudogene_subtype"
        ),
        "note": first_attribute(feature, "Note", "note", "description"),
        "annotation_feature_type": feature.feature_type,
        "annotation_source": feature.source,
        "annotation_line": feature.line_number,
        "classification_evidence": status.classification_evidence,
        "original_attributes": feature.attributes_raw,
    }


def exclusion_row(
    spec: SpeciesSpec,
    record_type: str,
    record_id: str,
    seqid: str,
    reason: str,
    compartment: str,
    details: str = "",
    count: int | str = 1,
    unit: str = "records",
) -> dict[str, object]:
    return {
        "species": spec.scientific_name,
        "record_type": record_type,
        "record_id": record_id,
        "seqid": seqid,
        "compartment": compartment,
        "reason": reason,
        "count": count,
        "unit": unit,
        "details": details,
    }


def mapping_summary(mappings: Sequence[PgbMapping]) -> dict[str, object]:
    total = len(mappings)
    mapped = sum(mapping.mapped for mapping in mappings)
    compartments = Counter(mapping.compartment for mapping in mappings if mapping.mapped)
    return {
        "pgb_total_genes": total,
        "pgb_mapped_genes": mapped,
        "pgb_unmapped_genes": total - mapped,
        "pgb_mapping_coverage": mapped / total if total else 0.0,
        "pgb_nuclear_genes": compartments["nuclear"],
        "pgb_chloroplast_genes": compartments["chloroplast"],
        "pgb_mitochondrial_genes": compartments["mitochondrial"],
        "pgb_unknown_genes": compartments["unknown"],
    }


def selected_statuses_with_cap(
    statuses: Sequence[CandidateStatus],
    maximum: int,
    seed: int,
    spec: SpeciesSpec,
    include_organelles: bool,
    include_unknown: bool,
) -> tuple[list[CandidateStatus], set[int]]:
    groups: list[tuple[str, list[CandidateStatus]]] = [
        ("nuclear", [status for status in statuses if status.eligible and status.compartment == "nuclear"])
    ]
    if include_organelles:
        groups.append(
            (
                "organellar",
                [
                    status
                    for status in statuses
                    if status.eligible and status.compartment in {"chloroplast", "mitochondrial"}
                ],
            )
        )
    if include_unknown:
        groups.append(
            ("unknown", [status for status in statuses if status.eligible and status.compartment == "unknown"])
        )
    selected: list[CandidateStatus] = []
    selected_ids: set[int] = set()
    for group_name, group in groups:
        capped = deterministic_cap(group, maximum, seed, spec.slug, "pseudogene", group_name)
        for status in capped:
            selected.append(status)
            selected_ids.add(id(status))
    return selected, selected_ids


def _intergenic_group(
    spec: SpeciesSpec,
    fasta: object,
    classifications: Mapping[str, SequenceClassification],
    masks: Mapping[str, Sequence[Interval]],
    compartments: set[str],
    maximum: int,
    seed: int,
    pgb_metrics: Sequence[tuple[float, float]],
) -> tuple[list[IntergenicCandidate], int, int, int]:
    intervals, total_nt, capacity, stranded_nt = calculate_availability_for_compartments(
        fasta, classifications, masks, compartments
    )
    candidates = sample_intergenic_candidates(
        spec,
        fasta,
        classifications,
        intervals,
        min(maximum, capacity),
        stable_seed(seed, "+".join(sorted(compartments))),
        pgb_metrics,
    )
    return candidates, total_nt, capacity, stranded_nt


def process_species(
    inputs: SpeciesInputs,
    pgb_fastas: Sequence[Path],
    output_root: Path,
    args: argparse.Namespace,
) -> SpeciesResult:
    spec = inputs.spec
    species_dir = output_root / spec.slug
    species_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for stale_name in (
            "included_non_nuclear_pseudogenes.fasta",
            "included_non_nuclear_pseudogenes.tsv",
            "included_non_nuclear_intergenic.fasta",
            "included_non_nuclear_intergenic.tsv",
        ):
            stale_path = species_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
    fasta = IndexedFasta(inputs.fasta, output_root / ".indexes" / f"{spec.slug}.fai")
    exclusions: list[dict[str, object]] = []
    try:
        pseudogenes, masks, genes, gff_hints = load_annotation(inputs.annotation, args.gene_buffer)
        reports = report_classifications(inputs.reports)
        classifications = classify_reference_sequences(fasta, reports, gff_hints)
        _, missing_mask_seqids = map_masks_to_reference(fasta, masks)
        for missing_seqid in missing_mask_seqids:
            exclusions.append(
                exclusion_row(
                    spec,
                    "annotation_aggregate",
                    "",
                    missing_seqid,
                    "annotation_seqid_absent_from_reference",
                    "unknown",
                    "mask features on this annotation sequence could not be applied",
                )
            )

        pgb_records, pgb_warnings = load_pgb_records(pgb_fastas)
        pgb_mappings, pgb_metrics = audit_pgb(
            pgb_records,
            genes,
            fasta,
            classifications,
            args.exact_sequence_mapping,
        )
        pgb_counts = mapping_summary(pgb_mappings)

        statuses = prepare_pseudogene_candidates(
            pseudogenes, fasta, classifications, args.pad_boundaries
        )
        selected_statuses, selected_status_ids = selected_statuses_with_cap(
            statuses,
            args.max_pseudogenes,
            args.seed,
            spec,
            args.include_organelles,
            args.include_unknown,
        )
        pseudogene_output: list[WindowRecord] = []
        extraction_failures: set[int] = set()
        emitted_status_ids: set[int] = set()
        status_exclusion_reasons: dict[int, str] = {}
        for status in selected_statuses:
            try:
                pseudogene_output.append(pseudogene_record(spec, status, fasta, args.pad_boundaries))
                emitted_status_ids.add(id(status))
            except (BuildError, KeyError, ValueError) as error:
                extraction_failures.add(id(status))
                status_exclusion_reasons[id(status)] = "sequence_extraction_failure"
                exclusions.append(
                    exclusion_row(
                        spec,
                        "pseudogene",
                        canonical_feature_id(status.feature),
                        status.feature.seqid,
                        "sequence_extraction_failure",
                        status.compartment,
                        str(error),
                    )
                )

        for status in statuses:
            if id(status) in extraction_failures:
                continue
            if not status.eligible:
                reason = status.exclusion_reason
            elif not compartment_is_selected(
                status.compartment, args.include_organelles, args.include_unknown
            ):
                reason = f"excluded_{status.compartment}_compartment"
            elif id(status) not in selected_status_ids:
                reason = "pseudogene_cap"
            else:
                continue
            status_exclusion_reasons[id(status)] = reason
            exclusions.append(
                exclusion_row(
                    spec,
                    "pseudogene",
                    canonical_feature_id(status.feature),
                    status.feature.seqid,
                    reason,
                    status.compartment,
                )
            )

        nuclear_candidates, eligible_nt, intergenic_capacity, stranded_nt = _intergenic_group(
            spec,
            fasta,
            classifications,
            masks,
            {"nuclear"},
            args.max_intergenic,
            args.seed,
            pgb_metrics,
        )
        reference_nt_by_compartment: Counter[str] = Counter()
        for reference_seqid in fasta.keys:
            classification = classifications.get(
                reference_seqid,
                SequenceClassification("unknown", "no classification evidence"),
            )
            reference_nt_by_compartment[classification.compartment] += fasta.length(reference_seqid)
        masked_and_buffered_nt = reference_nt_by_compartment["nuclear"] - eligible_nt
        if masked_and_buffered_nt:
            exclusions.append(
                exclusion_row(
                    spec,
                    "intergenic_aggregate",
                    "",
                    "",
                    "annotated_feature_mask_and_gene_buffer",
                    "nuclear",
                    "nuclear reference nucleotides removed by the expanded annotation mask",
                    masked_and_buffered_nt,
                    "bp",
                )
            )
        other_intergenic: list[IntergenicCandidate] = []
        extra_availability: dict[str, dict[str, int]] = {}
        if args.include_organelles:
            candidates, nt, capacity, stranded = _intergenic_group(
                spec,
                fasta,
                classifications,
                masks,
                {"chloroplast", "mitochondrial"},
                args.max_intergenic,
                args.seed,
                pgb_metrics,
            )
            other_intergenic.extend(candidates)
            extra_availability["organellar"] = {
                "eligible_nt": nt,
                "capacity": capacity,
                "stranded_nt": stranded,
            }
        if args.include_unknown:
            candidates, nt, capacity, stranded = _intergenic_group(
                spec,
                fasta,
                classifications,
                masks,
                {"unknown"},
                args.max_intergenic,
                args.seed,
                pgb_metrics,
            )
            other_intergenic.extend(candidates)
            extra_availability["unknown"] = {
                "eligible_nt": nt,
                "capacity": capacity,
                "stranded_nt": stranded,
            }
        nearest_index = NearestGeneIndex(genes, fasta.resolve)
        intergenic_output = intergenic_records(
            spec, [*nuclear_candidates, *other_intergenic], nearest_index
        )

        if intergenic_capacity > args.max_intergenic:
            exclusions.append(
                exclusion_row(
                    spec,
                    "intergenic_aggregate",
                    "",
                    "",
                    "intergenic_cap",
                    "nuclear",
                    "eligible nonoverlapping windows not selected",
                    intergenic_capacity - args.max_intergenic,
                    "windows",
                )
            )
        if stranded_nt:
            exclusions.append(
                exclusion_row(
                    spec,
                    "intergenic_aggregate",
                    "",
                    "",
                    "eligible_interval_remainder_shorter_than_6000bp",
                    "nuclear",
                    "eligible nucleotides outside the fixed nonoverlapping packing grid",
                    stranded_nt,
                    "bp",
                )
            )

        organellar_rows = []
        unknown_rows = []
        for status in statuses:
            emitted = id(status) in emitted_status_ids
            reason = "" if emitted else status_exclusion_reasons.get(
                id(status), status.exclusion_reason
            )
            row = pseudogene_audit_row(status, emitted, str(reason))
            row["species"] = spec.scientific_name
            row["dataset_id"] = make_dataset_id(
                spec,
                "pseudogene",
                (
                    f"{canonical_feature_id(status.feature)}_{status.feature.seqid}_"
                    f"{status.feature.start + 1}_{status.feature.end}_{status.feature.strand}"
                ),
            )
            if status.compartment in {"chloroplast", "mitochondrial"}:
                organellar_rows.append(row)
            elif status.compartment == "unknown":
                unknown_rows.append(row)

        nuclear_pseudogene_records = [
            record for record in pseudogene_output if record.compartment == "nuclear"
        ]
        non_nuclear_pseudogene_records = [
            record for record in pseudogene_output if record.compartment != "nuclear"
        ]
        nuclear_intergenic_records = [
            record for record in intergenic_output if record.compartment == "nuclear"
        ]
        non_nuclear_intergenic_records = [
            record for record in intergenic_output if record.compartment != "nuclear"
        ]

        write_fasta(species_dir / "nuclear_pseudogenes.fasta", nuclear_pseudogene_records)
        write_tsv(
            species_dir / "nuclear_pseudogenes.tsv",
            [window_row(record) for record in nuclear_pseudogene_records],
            [*BASE_METADATA_COLUMNS, *PSEUDOGENE_EXTRA_COLUMNS],
        )
        write_tsv(
            species_dir / "organellar_pseudogenes.tsv",
            organellar_rows,
            [*BASE_METADATA_COLUMNS, *PSEUDOGENE_EXTRA_COLUMNS],
        )
        write_tsv(
            species_dir / "unknown_pseudogenes.tsv",
            unknown_rows,
            [*BASE_METADATA_COLUMNS, *PSEUDOGENE_EXTRA_COLUMNS],
        )
        write_fasta(species_dir / "nuclear_intergenic.fasta", nuclear_intergenic_records)
        write_tsv(
            species_dir / "nuclear_intergenic.tsv",
            [window_row(record) for record in nuclear_intergenic_records],
            [*BASE_METADATA_COLUMNS, *INTERGENIC_EXTRA_COLUMNS],
        )
        if non_nuclear_pseudogene_records:
            write_fasta(
                species_dir / "included_non_nuclear_pseudogenes.fasta",
                non_nuclear_pseudogene_records,
            )
            write_tsv(
                species_dir / "included_non_nuclear_pseudogenes.tsv",
                [window_row(record) for record in non_nuclear_pseudogene_records],
                [*BASE_METADATA_COLUMNS, *PSEUDOGENE_EXTRA_COLUMNS],
            )
        if non_nuclear_intergenic_records:
            write_fasta(
                species_dir / "included_non_nuclear_intergenic.fasta",
                non_nuclear_intergenic_records,
            )
            write_tsv(
                species_dir / "included_non_nuclear_intergenic.tsv",
                [window_row(record) for record in non_nuclear_intergenic_records],
                [*BASE_METADATA_COLUMNS, *INTERGENIC_EXTRA_COLUMNS],
            )

        mapping_rows = [asdict(mapping) for mapping in pgb_mappings]
        write_tsv(
            species_dir / "pgb_compartment_audit.tsv",
            mapping_rows,
            ["gene_id", "mapped", "compartment", "evidence", "annotation_id", "seqid", "note"],
        )
        write_tsv(
            species_dir / "excluded.tsv",
            exclusions,
            [
                "species",
                "record_type",
                "record_id",
                "seqid",
                "compartment",
                "reason",
                "count",
                "unit",
                "details",
            ],
        )

        compartment_candidate_counts = Counter(status.compartment for status in statuses)
        nuclear_available = sum(
            status.eligible and status.compartment == "nuclear" for status in statuses
        )
        exclusion_counts: Counter[str] = Counter()
        for row in exclusions:
            raw_count = row.get("count", 1)
            count = int(raw_count) if str(raw_count).isdigit() else 1
            label = f"{row['reason']} [{row.get('unit', 'records')}]"
            exclusion_counts[label] += count
        summary: dict[str, object] = {
            "species": spec.scientific_name,
            "total_pseudogene_candidates": len(statuses),
            "nuclear_pseudogenes_available": nuclear_available,
            "nuclear_pseudogenes_emitted": len(nuclear_pseudogene_records),
            "chloroplast_pseudogenes": compartment_candidate_counts["chloroplast"],
            "mitochondrial_pseudogenes": compartment_candidate_counts["mitochondrial"],
            "unknown_pseudogenes": compartment_candidate_counts["unknown"],
            "eligible_intergenic_nt": eligible_nt,
            "nonoverlapping_intergenic_capacity": intergenic_capacity,
            "nuclear_intergenic_emitted": len(nuclear_intergenic_records),
            "pseudogene_cap": args.max_pseudogenes,
            "intergenic_cap": args.max_intergenic,
            **pgb_counts,
        }
        all_sequences = [*pseudogene_output, *intergenic_output]
        exact_lengths = all(len(record.sequence) == WINDOW_LENGTH for record in all_sequences)
        unique_ids = len({record.dataset_id for record in all_sequences}) == len(all_sequences)
        qc_failures = validate_emitted_records(pseudogene_output, intergenic_output)
        if len(nuclear_pseudogene_records) > args.max_pseudogenes:
            qc_failures.append("nuclear pseudogene output exceeds --max-pseudogenes")
        if len(nuclear_intergenic_records) > args.max_intergenic:
            qc_failures.append("nuclear intergenic output exceeds --max-intergenic")
        organellar_ids = [
            mapping.gene_id
            for mapping in pgb_mappings
            if mapping.mapped and mapping.compartment in {"chloroplast", "mitochondrial"}
        ]
        coverage = float(pgb_counts["pgb_mapping_coverage"])
        if organellar_ids:
            pgb_conclusion = "organellar_loci_detected"
        elif coverage == 1.0 and pgb_mappings:
            pgb_conclusion = "no_organellar_loci_detected_with_complete_mapping"
        else:
            pgb_conclusion = "no_mapped_organellar_loci_but_mapping_is_incomplete"
        qc: dict[str, object] = {
            "script_version": SCRIPT_VERSION,
            "species": spec.scientific_name,
            "resolved_inputs": {
                "directory": str(inputs.directory),
                "fasta": str(inputs.fasta),
                "annotation": str(inputs.annotation),
                "reports": [str(path) for path in inputs.reports],
                "resolution_evidence": list(inputs.resolution_evidence),
                "pgb_fastas": [str(path) for path in pgb_fastas],
            },
            "parameters": {
                "window_length": WINDOW_LENGTH,
                "anchor_offset": ANCHOR_OFFSET,
                "gene_buffer": args.gene_buffer,
                "max_pseudogenes": args.max_pseudogenes,
                "max_intergenic": args.max_intergenic,
                "seed": args.seed,
                "pad_boundaries": args.pad_boundaries,
                "include_organelles": args.include_organelles,
                "include_unknown": args.include_unknown,
                "exact_sequence_mapping": args.exact_sequence_mapping,
            },
            "summary": summary,
            "reference_compartments": dict(
                Counter(classification.compartment for classification in classifications.values())
            ),
            "reference_nt_by_compartment": dict(reference_nt_by_compartment),
            "intergenic_masked_and_buffered_nt": masked_and_buffered_nt,
            "intergenic_interval_remainder_nt": stranded_nt,
            "override_intergenic_availability": extra_availability,
            "exclusions_by_reason": dict(sorted(exclusion_counts.items())),
            "pgb_warnings": pgb_warnings,
            "pgb_compartment_conclusion": pgb_conclusion,
            "pgb_organellar_ids": organellar_ids,
            "all_emitted_sequences_exactly_6000bp": exact_lengths,
            "all_dataset_ids_unique_within_species": unique_ids,
            "qc_failures": qc_failures,
        }
        with (species_dir / "qc.json").open("w", encoding="utf-8") as handle:
            json.dump(qc, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if qc_failures:
            raise BuildError(f"QC failed for {spec.scientific_name}: {qc['qc_failures']}")
        return SpeciesResult(
            spec,
            inputs,
            summary,
            pseudogene_output,
            intergenic_output,
            pgb_mappings,
            exclusions,
            qc,
        )
    finally:
        fasta.close()


SUMMARY_COLUMNS = [
    "species",
    "total_pseudogene_candidates",
    "nuclear_pseudogenes_available",
    "nuclear_pseudogenes_emitted",
    "chloroplast_pseudogenes",
    "mitochondrial_pseudogenes",
    "unknown_pseudogenes",
    "eligible_intergenic_nt",
    "nonoverlapping_intergenic_capacity",
    "nuclear_intergenic_emitted",
    "pseudogene_cap",
    "intergenic_cap",
    "pgb_total_genes",
    "pgb_mapped_genes",
    "pgb_unmapped_genes",
    "pgb_mapping_coverage",
    "pgb_nuclear_genes",
    "pgb_chloroplast_genes",
    "pgb_mitochondrial_genes",
    "pgb_unknown_genes",
]


def write_combined_outputs(output_root: Path, results: Sequence[SpeciesResult]) -> None:
    nuclear_pseudogenes = [
        record
        for result in results
        for record in result.pseudogene_records
        if record.compartment == "nuclear"
    ]
    nuclear_intergenic = [
        record
        for result in results
        for record in result.intergenic_records
        if record.compartment == "nuclear"
    ]
    write_fasta(output_root / "all_species_nuclear_pseudogenes.fasta", nuclear_pseudogenes)
    write_tsv(
        output_root / "all_species_nuclear_pseudogenes.tsv",
        [window_row(record) for record in nuclear_pseudogenes],
        [*BASE_METADATA_COLUMNS, *PSEUDOGENE_EXTRA_COLUMNS],
    )
    write_fasta(output_root / "all_species_nuclear_intergenic.fasta", nuclear_intergenic)
    write_tsv(
        output_root / "all_species_nuclear_intergenic.tsv",
        [window_row(record) for record in nuclear_intergenic],
        [*BASE_METADATA_COLUMNS, *INTERGENIC_EXTRA_COLUMNS],
    )
    write_tsv(
        output_root / "summary.tsv",
        [result.summary for result in results],
        SUMMARY_COLUMNS,
    )

    pgb_rows: list[dict[str, object]] = []
    for result in results:
        organellar = [
            mapping
            for mapping in result.pgb_mappings
            if mapping.mapped and mapping.compartment in {"chloroplast", "mitochondrial"}
        ]
        pgb_rows.append(
            {
                "species": result.spec.scientific_name,
                "pgb_total_genes": result.summary["pgb_total_genes"],
                "pgb_mapped_genes": result.summary["pgb_mapped_genes"],
                "pgb_unmapped_genes": result.summary["pgb_unmapped_genes"],
                "pgb_mapping_coverage": result.summary["pgb_mapping_coverage"],
                "pgb_nuclear_genes": result.summary["pgb_nuclear_genes"],
                "pgb_chloroplast_genes": result.summary["pgb_chloroplast_genes"],
                "pgb_mitochondrial_genes": result.summary["pgb_mitochondrial_genes"],
                "pgb_unknown_genes": result.summary["pgb_unknown_genes"],
                "organellar_pgb_ids": ",".join(mapping.gene_id for mapping in organellar),
                "organellar_mapping_evidence": ";".join(
                    f"{mapping.gene_id}:{mapping.evidence}:{mapping.seqid}" for mapping in organellar
                ),
                "conclusion": result.qc["pgb_compartment_conclusion"],
            }
        )
    write_tsv(
        output_root / "pgb_compartment_summary.tsv",
        pgb_rows,
        [
            "species",
            "pgb_total_genes",
            "pgb_mapped_genes",
            "pgb_unmapped_genes",
            "pgb_mapping_coverage",
            "pgb_nuclear_genes",
            "pgb_chloroplast_genes",
            "pgb_mitochondrial_genes",
            "pgb_unknown_genes",
            "organellar_pgb_ids",
            "organellar_mapping_evidence",
            "conclusion",
        ],
    )


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in string_rows
    ]
    return "\n".join((header_line, separator, *body))


def reproduction_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/build_supplemental_loci.py",
        "--genomes-root",
        str(args.genomes_root),
        "--output",
        str(args.output),
        "--pgb-dir",
        str(args.pgb_dir),
        "--max-pseudogenes",
        str(args.max_pseudogenes),
        "--max-intergenic",
        str(args.max_intergenic),
        "--gene-buffer",
        str(args.gene_buffer),
        "--seed",
        str(args.seed),
    ]
    for enabled, flag in (
        (args.pad_boundaries, "--pad-boundaries"),
        (args.include_organelles, "--include-organelles"),
        (args.include_unknown, "--include-unknown"),
        (args.exact_sequence_mapping, "--exact-sequence-mapping"),
        (args.force, "--force"),
    ):
        if enabled:
            parts.append(flag)
    return " ".join(shlex.quote(part) for part in parts)


def print_completion(results: Sequence[SpeciesResult], command: str) -> None:
    print("\nResolved NCBI species directories:")
    for result in results:
        evidence = "; ".join(result.inputs.resolution_evidence)
        print(f"  {result.spec.scientific_name}: {result.inputs.directory} [{evidence}]")

    print("\nAvailable and emitted loci:")
    locus_rows = []
    for result in results:
        summary = result.summary
        locus_rows.append(
            (
                result.spec.slug,
                summary["total_pseudogene_candidates"],
                summary["nuclear_pseudogenes_available"],
                summary["nuclear_pseudogenes_emitted"],
                summary["chloroplast_pseudogenes"],
                summary["mitochondrial_pseudogenes"],
                summary["unknown_pseudogenes"],
                summary["eligible_intergenic_nt"],
                summary["nonoverlapping_intergenic_capacity"],
                summary["nuclear_intergenic_emitted"],
            )
        )
    print(
        format_table(
            (
                "species",
                "pseudo total",
                "pseudo avail",
                "pseudo emit",
                "cp pseudo",
                "mt pseudo",
                "unk pseudo",
                "intergenic nt",
                "intergenic cap.",
                "intergenic emit",
            ),
            locus_rows,
        )
    )

    print("\nPGB compartment audit:")
    pgb_rows = []
    for result in results:
        summary = result.summary
        pgb_rows.append(
            (
                result.spec.slug,
                summary["pgb_total_genes"],
                summary["pgb_mapped_genes"],
                f"{float(summary['pgb_mapping_coverage']):.1%}",
                summary["pgb_nuclear_genes"],
                summary["pgb_chloroplast_genes"],
                summary["pgb_mitochondrial_genes"],
                summary["pgb_unknown_genes"],
            )
        )
    print(
        format_table(
            ("species", "total", "mapped", "coverage", "nuclear", "cp", "mt", "unknown"),
            pgb_rows,
        )
    )
    for result in results:
        organellar = [
            mapping
            for mapping in result.pgb_mappings
            if mapping.mapped and mapping.compartment in {"chloroplast", "mitochondrial"}
        ]
        if organellar:
            print(f"  {result.spec.slug} organellar PGB mappings:")
            for mapping in organellar:
                print(
                    f"    {mapping.gene_id}: {mapping.compartment}; "
                    f"{mapping.evidence}; {mapping.seqid}"
                )
        else:
            print(
                f"  {result.spec.slug}: {result.qc['pgb_compartment_conclusion']} "
                f"(coverage {float(result.summary['pgb_mapping_coverage']):.1%})"
            )

    print("\nExclusions and QC:")
    for result in results:
        exclusions = result.qc["exclusions_by_reason"]
        display = ", ".join(f"{key}={value}" for key, value in exclusions.items()) or "none"
        failures = result.qc["qc_failures"] or "none"
        print(f"  {result.spec.slug}: exclusions [{display}]; QC failures [{failures}]")
    print("  Confirmed: every emitted sequence is exactly 6,000 bp.")
    print(f"\nReproduce with:\n{command}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create indexed-FASTA pseudogene/intergenic supplements and audit PGB compartments."
    )
    parser.add_argument("--genomes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pgb-dir", type=Path, required=True)
    parser.add_argument("--max-pseudogenes", type=int, default=5_000)
    parser.add_argument("--max-intergenic", type=int, default=5_000)
    parser.add_argument("--gene-buffer", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pad-boundaries",
        action="store_true",
        help="Pad reference-boundary windows with N instead of excluding them.",
    )
    parser.add_argument(
        "--include-organelles",
        action="store_true",
        help="Additionally emit capped chloroplast/mitochondrial sets in clearly named non-nuclear files.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Additionally emit capped unknown-compartment sets in clearly named non-nuclear files.",
    )
    parser.add_argument(
        "--exact-sequence-mapping",
        action="store_true",
        help="For unmapped PGB IDs, hash exact oriented 6 kb gene windows as a separate fallback.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow generated files in an existing nonempty output directory to be overwritten.",
    )
    args = parser.parse_args(argv)
    for option in ("max_pseudogenes", "max_intergenic", "gene_buffer"):
        if getattr(args, option) < 0:
            parser.error(f"--{option.replace('_', '-')} must be nonnegative")
    return args


def build(args: argparse.Namespace) -> list[SpeciesResult]:
    # Keep symlink components visible in diagnostics and in the reproduction
    # command; resolve(strict=False) would erase a broken shared-link boundary.
    args.genomes_root = Path(os.path.abspath(args.genomes_root.expanduser()))
    args.output = Path(os.path.abspath(args.output.expanduser()))
    args.pgb_dir = Path(os.path.abspath(args.pgb_dir.expanduser()))
    resolved = discover_species_inputs(args.genomes_root)
    pgb_fastas = discover_pgb_fastas(args.pgb_dir)
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise BuildError(
            f"Output directory is nonempty: {args.output}. Choose a new directory or pass --force."
        )
    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise BuildError(f"Could not create output directory {args.output}: {error}") from error
    results: list[SpeciesResult] = []
    for spec in SPECIES:
        print(f"Processing {spec.scientific_name}...", flush=True)
        results.append(
            process_species(resolved[spec.slug], pgb_fastas[spec.slug], args.output, args)
        )
    write_combined_outputs(args.output, results)
    print_completion(results, reproduction_command(args))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build(args)
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: filesystem operation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
