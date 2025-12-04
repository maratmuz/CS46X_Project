import random
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
from collections import namedtuple


# Named tuple for gene annotation
GeneAnnotation = namedtuple('GeneAnnotation', [
    'seq_id', 'start', 'end', 'strand', 'gene_id', 'cds_features'
])
# CDS feature for translation
CDSFeature = namedtuple('CDSFeature', ['start', 'end', 'phase'])

class GenomicDataLoader:
    def __init__(self):
        self.supported_formats = [
            'fasta',
            'gff3',
        ]
        self._selected_samples = None
        self._sample_index = 0
        self._genes = []  # List of GeneAnnotation tuples
        self._gene_index = 0

    def load(self, path, format, gff_path=None, verbose=False):
        """
        Load genomic data from file(s).
        
        Args:
            path: Path to FASTA file (required for 'fasta' format, optional for 'gff3')
            format: Format type ('fasta' or 'gff3')
            gff_path: Optional path to GFF3 file (used when loading FASTA + GFF separately)
            verbose: Print detailed information about loaded data
        """
        if format.lower() not in self.supported_formats:
            raise ValueError(
                f'Format "{format}" was not found in the list of supported data types.'
            )
        elif format.lower() == 'fasta':
            self._data = list(SeqIO.parse(path, "fasta"))
            
            # If GFF path is provided, load annotations separately
            if gff_path:
                self._genes = self._load_gff3_annotations(gff_path, verbose=False)
                if verbose:
                    self._print_loaded_data(path, gff_path)
            else:
                if verbose:
                    self._print_loaded_fasta(path)
        
        elif format.lower() == 'gff3':
            # For gff3 format, path can be GFF file (with embedded or separate FASTA)
            if gff_path:
                # Both paths provided: use path as FASTA, gff_path as GFF
                self._data = list(SeqIO.parse(path, "fasta"))
                self._genes = self._load_gff3_annotations(gff_path, verbose=False)
                if verbose:
                    self._print_loaded_data(path, gff_path)
            else:
                # Only GFF path provided, try embedded FASTA or separate file
                self._data, self._genes = self._load_gff3(path, verbose)

    def read(self, splits):
        sampled_seq = random.choice(self._data)
        seq_len = len(sampled_seq)
        req_len = sum(splits)

        seq_start = random.randint(0, seq_len - req_len)

        data = []

        for split in splits:
            split_data = sampled_seq[seq_start : seq_start + split]
            seq_start += split
            data.append(str(split_data.seq))

        return data

    def initialize_unique_samples(self, num_samples=4):
        """
        Initialize a pool of unique samples to be used across multiple reads.
        Call this before starting a new test to select unique samples.
        
        Args:
            num_samples: Number of unique samples to randomly select (default: 4)
        """
        if num_samples > len(self._data):
            raise ValueError(
                f"Cannot select {num_samples} samples from {len(self._data)} available sequences"
            )
        
        self._selected_samples = random.sample(self._data, num_samples)
        self._sample_index = 0

    def read_start(self, splits):
        """
        Read from the pool of unique samples starting from position 0.
        Cycles through the unique samples selected by initialize_unique_samples().
        
        Args:
            splits: List of split lengths to extract from each sequence
        
        Returns:
            List of strings corresponding to the requested splits
        """
        if self._selected_samples is None:
            raise RuntimeError(
                "Must call initialize_unique_samples() before using read_start()"
            )
        
        # Get the next sample, cycle through available samples
        sampled_seq = self._selected_samples[self._sample_index % len(self._selected_samples)]
        self._sample_index += 1
        
        seq_len = len(sampled_seq)
        req_len = sum(splits) # the test's specified seq length + pred length
        
        if req_len > seq_len:
            raise ValueError(
                f"Required length {req_len} exceeds sequence length {seq_len}"
            )
        
        seq_start = 0
        data = []
        
        for split in splits:
            split_data = sampled_seq[seq_start : seq_start + split]
            seq_start += split
            data.append(str(split_data.seq))
        
        return data

    def read_midpoint(self, pred_len):
        """
        Read from the pool of unique samples using midpoint split.
        Splits sequence at midpoint (50%) for prompt and target, matching test_evo2_generation.py.
        Cycles through the unique samples selected by initialize_unique_samples().
        
        Args:
            pred_len: Length of the prediction/target sequence to extract after midpoint
        
        Returns:
            Tuple of (prompt, target) strings
        """
        if self._selected_samples is None:
            raise RuntimeError(
                "Must call initialize_unique_samples() before using read_midpoint()"
            )

        sampled_seq = self._selected_samples[self._sample_index % len(self._selected_samples)]
        self._sample_index += 1
        
        seq_len = len(sampled_seq)
        mid_point = seq_len // 2
        
        # Check if we have enough sequence after midpoint for prediction
        if mid_point + pred_len > seq_len:
            raise ValueError(
                f"Sequence length {seq_len} is too short for midpoint {mid_point} + pred_len {pred_len}"
            )
        
        prompt = str(sampled_seq[:mid_point].seq)
        target = str(sampled_seq[mid_point:mid_point + pred_len].seq)
        
        return prompt, target

    def _load_gff3_annotations(self, gff_path, verbose=False):
        """
        Load only GFF3 annotations without sequences.
        Extracts gene features (whole genes) and associates CDS features for translation.
        
        Args:
            gff_path: Path to GFF3 file
            verbose: Print detailed information
        
        Returns:
            List of GeneAnnotation tuples with associated CDS features
        """
        genes_dict = {}  # gene_id -> GeneAnnotation
        cds_by_gene = {}  # gene_id -> list of CDS features
        mrna_to_gene = {}  # mRNA/transcript ID -> gene ID (for TAIR10 structure)
        
        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if line.startswith('##FASTA'):
                    break
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                seq_id = fields[0]
                feature_type = fields[2]
                start = int(fields[3]) - 1  # Convert to 0-based
                end = int(fields[4])  # End is exclusive in 0-based
                strand = fields[6]
                phase = fields[7] if len(fields) > 7 else '0'
                
                # Parse attributes
                attributes = {}
                for attr in fields[8].split(';'):
                    if '=' in attr:
                        key, value = attr.split('=', 1)
                        attributes[key] = value
                
                # Process gene features
                if feature_type == 'gene':
                    gene_id = attributes.get('ID', f"{seq_id}_{start}_{end}")
                    genes_dict[gene_id] = GeneAnnotation(
                        seq_id=seq_id,
                        start=start,
                        end=end,
                        strand=strand,
                        gene_id=gene_id,
                        cds_features=[]  # Will be populated from CDS features
                    )
                    if gene_id not in cds_by_gene:
                        cds_by_gene[gene_id] = []
                
                # Process mRNA/transcript features (map to genes)
                elif feature_type in ['mRNA', 'transcript']:
                    mrna_id = attributes.get('ID', None)
                    parent_gene_id = attributes.get('Parent', None)
                    
                    if mrna_id and parent_gene_id:
                        # Map mRNA ID to gene ID
                        mrna_to_gene[mrna_id] = parent_gene_id
                        # Handle comma-separated Parent values
                        if ',' in parent_gene_id:
                            parent_gene_id = parent_gene_id.split(',')[0].strip()
                            mrna_to_gene[mrna_id] = parent_gene_id
                
                # Process CDS features
                elif feature_type == 'CDS':
                    parent_id = attributes.get('Parent', None)
                    cds_id = attributes.get('ID', None)
                    
                    # Handle comma-separated Parent values (take first)
                    if parent_id and ',' in parent_id:
                        parent_id = parent_id.split(',')[0].strip()
                    
                    # Find the gene ID that this CDS belongs to
                    target_gene_id = None
                    
                    if parent_id:
                        # Check if parent is a gene ID
                        if parent_id in genes_dict:
                            target_gene_id = parent_id
                        # Check if parent is an mRNA/transcript ID (e.g., AT1G09040.1)
                        elif parent_id in mrna_to_gene:
                            target_gene_id = mrna_to_gene[parent_id]
                            # Ensure gene exists
                            if target_gene_id not in genes_dict:
                                # Gene should exist, but create it if missing
                                genes_dict[target_gene_id] = GeneAnnotation(
                                    seq_id=seq_id,
                                    start=start,
                                    end=end,
                                    strand=strand,
                                    gene_id=target_gene_id,
                                    cds_features=[]
                                )
                        else:
                            # Parent not found - might be a new gene or transcript without gene parent
                            # Try to extract gene ID by removing suffix (e.g., AT1G09040.1 -> AT1G09040)
                            if '.' in parent_id:
                                potential_gene_id = parent_id.rsplit('.', 1)[0]
                                if potential_gene_id in genes_dict:
                                    target_gene_id = potential_gene_id
                                else:
                                    # Use parent as gene ID (will create gene below)
                                    target_gene_id = parent_id
                            else:
                                target_gene_id = parent_id
                    
                    # Fallback: use CDS ID or create new
                    if target_gene_id is None:
                        if cds_id and cds_id in genes_dict:
                            target_gene_id = cds_id
                        else:
                            target_gene_id = cds_id or f"{seq_id}_{start}_{end}"
                    
                    # Ensure target_gene_id exists in genes_dict
                    if target_gene_id not in genes_dict:
                        # Create gene from CDS coordinates
                        genes_dict[target_gene_id] = GeneAnnotation(
                            seq_id=seq_id,
                            start=start,
                            end=end,
                            strand=strand,
                            gene_id=target_gene_id,
                            cds_features=[]
                        )
                    
                    # CRITICAL: Ensure cds_by_gene entry exists before appending
                    if target_gene_id not in cds_by_gene:
                        cds_by_gene[target_gene_id] = []
                    
                    # Add CDS to the gene
                    cds_by_gene[target_gene_id].append(CDSFeature(start=start, end=end, phase=phase))
        
        # Associate CDS features with genes
        genes = []
        for gene_id, gene_ann in genes_dict.items():
            cds_list = cds_by_gene.get(gene_id, gene_ann.cds_features)
            # Sort CDS features by start position
            cds_list.sort(key=lambda x: x.start)
            genes.append(GeneAnnotation(
                seq_id=gene_ann.seq_id,
                start=gene_ann.start,
                end=gene_ann.end,
                strand=gene_ann.strand,
                gene_id=gene_ann.gene_id,
                cds_features=cds_list
            ))
        
        return genes

    def _load_gff3(self, path, verbose=False):
        """
        Load GFF3 file and extract CDS features.
        GFF3 files can have FASTA sequences embedded after ##FASTA directive.
        If no FASTA is embedded, try to load from a corresponding .fa or .fasta file.
        
        Returns:
            Tuple of (sequences, genes) where sequences is list of SeqRecord objects
            and genes is list of GeneAnnotation tuples
        """
        sequences = []
        genes = []
        current_seq = None
        current_id = None
        seq_dict = {}
        in_fasta_section = False
        
        # First pass: load FASTA sequences if embedded
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('##FASTA'):
                    in_fasta_section = True
                    continue
                if in_fasta_section and line.startswith('>'):
                    # New sequence header
                    if current_seq and current_id:
                        seq_dict[current_id] = current_seq
                    current_id = line.strip()[1:].split()[0]
                    current_seq = ""
                elif in_fasta_section:
                    current_seq += line.strip()
            if current_seq and current_id:
                seq_dict[current_id] = current_seq
        
        # If no embedded FASTA, try to load from separate FASTA file
        if not seq_dict:
            fasta_path = Path(path).with_suffix('.fa')
            if not fasta_path.exists():
                fasta_path = Path(path).with_suffix('.fasta')
            if fasta_path.exists():
                sequences = list(SeqIO.parse(str(fasta_path), "fasta"))
            else:
                raise ValueError(f"No FASTA sequences found in GFF3 file {path} and no corresponding .fa/.fasta file found")
        else:
            # Convert to SeqRecord objects
            for seq_id, seq_str in seq_dict.items():
                sequences.append(SeqIO.SeqRecord(Seq(seq_str), id=seq_id))
        
        # Second pass: parse GFF3 features (extract gene features)
        genes = self._load_gff3_annotations(path, verbose=False)
        
        if verbose:
            print(f"\nLoaded {len(sequences)} sequences and {len(genes)} gene features from {path}")
            print("=" * 70)
            for idx, seq in enumerate(sequences, 1):
                seq_len = len(seq)
                gene_count = sum(1 for g in genes if g.seq_id == seq.id)
                print(f"  Sequence {idx}: {seq.id[:40]:40} | Length: {seq_len:8,} | CDS features: {gene_count:4}")
            print("=" * 70)
            print()
        
        return sequences, genes

    def _print_loaded_fasta(self, path):
        """Print information about loaded FASTA sequences."""
        print(f"\nLoaded {len(self._data)} sequences from {path}")
        print("=" * 70)
        for idx, seq in enumerate(self._data, 1):
            seq_len = len(seq)
            mid_point = seq_len // 2
            print(f"  Sample {idx}: {seq.id[:40]:40} | Length: {seq_len:8,} | Midpoint: {mid_point:8,}")
        print("=" * 70)
        print(f"Average sequence length: {sum(len(s) for s in self._data) / len(self._data):,.0f}")
        print()

    def _print_loaded_data(self, fasta_path, gff_path):
        """Print information about loaded FASTA and GFF data."""
        print(f"\nLoaded {len(self._data)} sequences from {fasta_path}")
        print(f"Loaded {len(self._genes)} genes from {gff_path}")
        print("=" * 70)
        
        # Group genes by sequence/chromosome
        seq_gene_counts = {}
        for gene in self._genes:
            seq_gene_counts[gene.seq_id] = seq_gene_counts.get(gene.seq_id, 0) + 1
        
        for idx, seq in enumerate(self._data, 1):
            seq_len = len(seq)
            gene_count = seq_gene_counts.get(seq.id, 0)
            print(f"  Sequence {idx}: {seq.id[:40]:40} | Length: {seq_len:8,} | Genes: {gene_count:4}")
        print("=" * 70)
        print()

    def initialize_gene_evaluation(self, num_genes=None, chromosomes=None, min_cds_length=None, 
                                   min_target_length=None, max_expected_min_recovery=None):
        """
        Initialize gene evaluation by selecting genes from the loaded GFF3 data.
        
        Args:
            num_genes: Number of genes to select (default: all available after filtering)
            chromosomes: Chromosome/sequence IDs to filter by. Can be:
                         - List: ['Chr1', 'Chr2']
                         - String: 'Chr1' (single chromosome)
                         - None: uses all chromosomes
            min_cds_length: Minimum CDS length required (genes shorter than this will be filtered out).
                           Typically set to seed length (500 for prokaryotes, 1000 for eukaryotes)
            min_target_length: Minimum target sequence length (CDS after seed) required.
                              Filters out genes where target is too short (avoids guaranteed 100% recovery).
                              Default: 3 bp (prevents 1-2 bp targets that can't form complete codons)
            max_expected_min_recovery: Maximum allowed expected minimum recovery percentage.
                                     Filters out genes where seed already covers most of protein.
                                     Default: 98.0 (exclude genes with >98% expected recovery)
        """
        if not self._genes:
            raise RuntimeError("No genes loaded. Load a GFF file first.")
        
        # Normalize chromosomes input (handle string, list, or OmegaConf ListConfig)
        if chromosomes is not None:
            if isinstance(chromosomes, str):
                chromosomes = [chromosomes]
            elif hasattr(chromosomes, '__iter__') and not isinstance(chromosomes, (str, bytes)):
                # Convert iterable (list, ListConfig, etc.) to list
                chromosomes = list(chromosomes)
            else:
                raise ValueError(f"chromosomes must be a string, iterable (list), or None, got {type(chromosomes)}")
        
        # Filter by chromosomes if specified
        filtered_genes = self._genes
        if chromosomes is not None:
            # Get available chromosome names from both sequences and genes
            available_seq_ids = {seq.id for seq in self._data}
            available_gene_seq_ids = {g.seq_id for g in self._genes}
            
            # Try multiple matching strategies
            def matches_chromosome(seq_id, chrom_list):
                """Check if seq_id matches any chromosome in chrom_list."""
                seq_id_lower = seq_id.lower()
                for chrom in chrom_list:
                    chrom_lower = chrom.lower()
                    # Exact match
                    if seq_id_lower == chrom_lower:
                        return True
                    # Starts with match
                    if seq_id_lower.startswith(chrom_lower):
                        return True
                    # Remove common prefixes and match
                    seq_id_clean = seq_id_lower.replace('chr', '').replace('chromosome', '').replace('_', '')
                    chrom_clean = chrom_lower.replace('chr', '').replace('chromosome', '').replace('_', '')
                    if seq_id_clean == chrom_clean:
                        return True
                return False
            
            filtered_genes = [
                g for g in self._genes 
                if matches_chromosome(g.seq_id, chromosomes)
            ]
            
            if not filtered_genes:
                available_chromosomes = sorted(set(g.seq_id for g in self._genes))
                raise ValueError(
                    f"No genes found for specified chromosomes {chromosomes}. "
                    f"Available chromosomes: {available_chromosomes}"
                )
        
        # Filter by minimum CDS length if specified
        genes_before_length_filter = len(filtered_genes)
        genes_filtered_out_by_length = 0
        genes_filtered_out_by_target = 0
        if min_cds_length is not None:
            filtered_genes_with_cds = []
            for gene in filtered_genes:
                # Calculate total CDS length by summing all CDS feature lengths
                total_cds_length = sum(cds.end - cds.start for cds in gene.cds_features)
                if total_cds_length > min_cds_length:
                    # Additional check: ensure target sequence (after seed) is at least 3 bp
                    # This prevents guaranteed 100% recovery from 1-2 bp targets (which can't form complete codons)
                    target_length = total_cds_length - min_cds_length
                    min_target = min_target_length if min_target_length is not None else 3
                    if target_length >= min_target:
                        filtered_genes_with_cds.append(gene)
                    else:
                        genes_filtered_out_by_target += 1
                else:
                    genes_filtered_out_by_length += 1
            
            filtered_genes = filtered_genes_with_cds
            genes_filtered_out = genes_before_length_filter - len(filtered_genes)
            
            if not filtered_genes:
                min_target_display = min_target_length if min_target_length is not None else 3
                raise ValueError(
                    f"No genes found with CDS length > {min_cds_length} bp AND target length >= {min_target_display} bp "
                    f"(after chromosome filtering). "
                    f"{genes_filtered_out_by_length} genes were too short, "
                    f"{genes_filtered_out_by_target} genes had target sequence too short."
                )
        
        # Select subset if num_genes specified
        if num_genes is None:
            self._selected_genes = filtered_genes
        else:
            if num_genes > len(filtered_genes):
                raise ValueError(
                    f"Cannot select {num_genes} genes from {len(filtered_genes)} available "
                    f"(after chromosome filtering). Use None to select all {len(filtered_genes)} genes."
                )
            self._selected_genes = random.sample(filtered_genes, num_genes)
        
        # Print summary
        selected_chromosomes = sorted(set(g.seq_id for g in self._selected_genes))
        print(f"\nInitialized gene evaluation:")
        print(f"  Selected chromosomes: {selected_chromosomes}")
        print(f"  Total genes selected: {len(self._selected_genes)}")
        if chromosomes is not None:
            print(f"  Requested chromosomes: {chromosomes}")
        if min_cds_length is not None:
            print(f"  Minimum CDS length filter: > {min_cds_length:,} bp")
            print(f"  Minimum target length filter: >= {min_target_length if min_target_length is not None else 3:,} bp")
            if genes_filtered_out_by_length > 0:
                print(f"  Genes filtered out (CDS too short): {genes_filtered_out_by_length}")
            if genes_filtered_out_by_target > 0:
                print(f"  Genes filtered out (target too short): {genes_filtered_out_by_target}")
        print()
        
        self._gene_index = 0

    def get_gene_prompt(self, organism_type='prokaryote', upstream_bp=1000):
        """
        Get a gene prediction prompt according to the paper specification.
        
        Paper: "For each gene, a prompt of 1,000 base pairs upstream of the gene 
        and the first 500 bp (prokaryotes) or 1000 bp (eukaryotes) of the gene sequence"
        
        Prompt construction:
        - 1,000 bp upstream of the gene (from gene start)
        - First 500 bp of gene sequence for prokaryotes/archaea/yeast 
        - First 1,000 bp of gene sequence for eukaryotes
        
        Args:
            organism_type: 'prokaryote' or 'eukaryote' (determines gene seed length)
            upstream_bp: Number of base pairs upstream to include (default: 1000)
        
        Returns:
            Tuple of (prompt_seq, target_gene_seq, full_cds_seq, gene_annotation)
            where:
            - prompt_seq: The complete prompt (upstream + gene seed)
            - target_gene_seq: The remaining gene sequence to be predicted
            - full_cds_seq: The full CDS sequence (for translation to protein)
            - gene_annotation: The GeneAnnotation tuple
        """
        if not hasattr(self, '_selected_genes') or not self._selected_genes:
            raise RuntimeError("Must call initialize_gene_evaluation() before using get_gene_prompt()")
        
        # Get next gene
        gene = self._selected_genes[self._gene_index % len(self._selected_genes)]
        self._gene_index += 1
        
        # Find the sequence for this gene
        seq_record = None
        for seq in self._data:
            if seq.id == gene.seq_id:
                seq_record = seq
                break
        
        if seq_record is None:
            raise ValueError(f"Sequence {gene.seq_id} not found in loaded sequences")
        
        seq_str = str(seq_record.seq).upper()
        seq_len = len(seq_str)
        
        # Determine gene seed length based on organism type
        # Paper: "the first 500 bp (prokaryotes) or 1000 bp (eukaryotes) of the gene sequence"
        if organism_type.lower() in ['prokaryote', 'archaea', 'yeast']:
            gene_seed_len = 500
        elif organism_type.lower() in ['eukaryote', 'eukaryotes']:
            gene_seed_len = 1000
        else:
            raise ValueError(f"Unknown organism_type: {organism_type}. Use 'prokaryote' or 'eukaryote'")
        
        # Extract gene coordinates for upstream region
        gene_start = gene.start
        gene_end = gene.end
        
        # Extract upstream region (upstream of the gene start)
        if gene.strand == '+':
            upstream_start = max(0, gene_start - upstream_bp)
            upstream_seq = seq_str[upstream_start:gene_start]
        else:
            # For reverse strand, upstream is after the gene end
            upstream_start = min(seq_len, gene_end)
            upstream_end = min(seq_len, gene_end + upstream_bp)
            upstream_seq = seq_str[upstream_start:upstream_end]
            upstream_seq = str(Seq(upstream_seq).reverse_complement())
        
        # Extract full CDS for translation (from cds_features)
        # Combine all CDS features to get the full coding sequence
        # CRITICAL: Handle phase correctly to preserve codon boundaries across exon junctions
        if not gene.cds_features:
            raise ValueError(f"Gene {gene.gene_id} has no CDS features. Cannot extract coding sequence.")
        
        # Sort CDS features by genomic position (in 5' to 3' order for coding sequence)
        # For forward strand: sort ascending (start positions increase 5'->3')
        # For reverse strand: sort descending (start positions decrease 5'->3' in coding direction)
        if gene.strand == '+':
            sorted_cds = sorted(gene.cds_features, key=lambda x: x.start)
        else:  # Reverse strand
            sorted_cds = sorted(gene.cds_features, key=lambda x: x.start, reverse=True)
        
        cds_parts = []
        for idx, cds in enumerate(sorted_cds):
            cds_seq = seq_str[cds.start:cds.end]
            
            # Phase trimming happens AFTER reverse complementing so it applies to the 5' end
            if gene.strand == '-':
                cds_seq = str(Seq(cds_seq).reverse_complement())
            
            # Parse phase: 0, 1, 2, or '.' (unknown/not applicable)
            try:
                phase = int(cds.phase) if str(cds.phase) != '.' else 0
            except (ValueError, TypeError, AttributeError):
                phase = 0
            
            if phase > 0:
                if len(cds_seq) > phase:
                    # Trim the first 'phase' nucleotides from the 5' end to maintain codon alignment
                    cds_seq = cds_seq[phase:]
                else:
                    # Edge case: CDS shorter than phase (shouldn't happen, but handle gracefully)
                    print(f"Warning: CDS {cds.start}-{cds.end} has phase {phase} but only {len(cds_seq)} bases. "
                          f"CDS too short for phase. Keeping all bases.")
            
            cds_parts.append(cds_seq)
        
        full_cds_seq = ''.join(cds_parts)
        
        # Extract CDS seed (first part of CDS sequence)
        cds_seed = full_cds_seq[:gene_seed_len]
        
        # Construct prompt: upstream + CDS seed (first part of CDS/gene sequence)
        prompt_seq = upstream_seq + cds_seed
        
        # Target is the remaining CDS sequence after the seed
        target_cds_seq = full_cds_seq[gene_seed_len:]
        
        return prompt_seq, target_cds_seq, full_cds_seq, gene

