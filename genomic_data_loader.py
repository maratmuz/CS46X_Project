import random
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import Align
from Bio import SeqIO
from BCBio import GFF


class GenomicDataLoader:
    def __init__(self):
        self.supported_formats = [
            'fasta',
            'gff3',
        ]
        self._selected_samples = None
        self._sample_index = 0

    def load(self, fasta_path, gff_path):
        '''
        takes in two data file paths (fa and gff)

        outputs a list of chromosones and sublists of genes for each chromosone
        the 'list of chromosones' is literally just apyhton list of lists where each of the sublists 
        represents the collection of all the genes that were parsed from that chromosone

        each of the elements in those sublists represents an individual gene and is literally just a string
        each of these strings should also start with 1000 bp of 'upstream context' eg the first 1000 bp / nucleotides in the 
        larger chromosone sequence before the gene itself started


        EXCLUDES the very first gene in each chromosone,
        as well as any genes which have 
        '''
        chromosomes = []


        seqs = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

        with open(gff_path) as handle:
            for i, rec in enumerate(GFF.parse(handle, base_dict=seqs)):
                chr_genes = []
                skipped_genes = 0

                for feature in rec.features:
                    if feature.type == 'gene':
                        # gene_start = int(feature.location.start)
                        # gene_end = int(feature.location.end)
                        # gene_seq = rec.seq[gene_start:gene_end]

                        gene_start = int(feature.location.start)
                        gene_end = int(feature.location.end)
                        ori_len = gene_end - gene_start
                        
                        # 2. Drill down to CDS to find the biological Start (ATG)
                        if feature.sub_features and feature.sub_features[0].type == 'mRNA':
                            mrna = feature.sub_features[0]
                            
                            # Collect all CDS chunks
                            all_cds = [f for f in mrna.sub_features if f.type == 'CDS']
                            
                            if all_cds:
                                # Sort by genomic position
                                all_cds.sort(key=lambda f: f.location.start)
                                
                                # CHECK STRAND ON THE LOCATION OBJECT
                                if feature.location.strand == 1:
                                    # Forward: Start is the lowest coordinate of the first CDS
                                    gene_start = int(all_cds[0].location.start)
                                    
                                elif feature.location.strand == -1:
                                    # Reverse: Start is the highest coordinate of the last CDS
                                    gene_end = int(all_cds[-1].location.end)

                        # 3. Slice the sequence
                        gene_seq = rec.seq[gene_start:gene_end]
                        
                        # 4. Handle Reverse Complement for reverse strand genes
                        # (Optional, but required if you want the string to actually start with "ATG")
                        if feature.location.strand == -1:
                            gene_seq = gene_seq.reverse_complement()

                        # Gene validation checks

                        # Is the gene at least 1,010 bp long 
                        # -> we typically give 1,000 bp of context, so it needs something left to predict ...
                        # -> the choice of 10 is arbitrary
                        # -> the 1,000 bp context is in addition to the 1,000 of upstream context
                        # -> so the total input length is typically 2,000
                        if len(gene_seq) <= 1010:
                            skipped_genes += 1
                            continue

                        # Does the gene have at least 1,000 bp of preceeding nucleotides (necessary for the upstream context provided to Evo2)
                        if gene_start <= 1000:
                            skipped_genes += 1
                            continue

                        upstream_context_1k = rec.seq[(gene_start - 1000):gene_start]

                        chr_genes.append({
                            'id': feature.id,
                            'type': feature.type,
                            'qualifiers': feature.qualifiers,
                            'seq': gene_seq,
                            'upstream_1k': upstream_context_1k,
                        })
                
                chromosomes.append(chr_genes)

        self._data = chromosomes
    
    def num_chromosomes(self):
        return len(self._data)

    def sample_genes(self, chr_id, gene_context_len=1000, count=1):
        genes = []
        chromosome = self._data[chr_id]

        for i in range(count):
            sampled_gene = random.choice(chromosome)
            gene_data = {
                'input': sampled_gene['upstream_1k'] + sampled_gene['seq'][:gene_context_len],
                'label': sampled_gene['seq'][gene_context_len:],
            }
            genes.append(gene_data)

        return genes

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
        mid_point = 2 * (seq_len // 4)
        
        # Check if we have enough sequence after midpoint for prediction
        if mid_point + pred_len > seq_len:
            raise ValueError(
                f"Sequence length {seq_len} is too short for midpoint {mid_point} + pred_len {pred_len}"
            )
        
        prompt = str(sampled_seq[:mid_point].seq)
        target = str(sampled_seq[mid_point:mid_point + pred_len].seq)
        
        return prompt, target

