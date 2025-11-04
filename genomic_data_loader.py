import random
from Bio import SeqIO


class GenomicDataLoader:
    def __init__(self):
        self.supported_formats = [
            'fasta',
            'gff3',
        ]
        self._selected_samples = None
        self._sample_index = 0

    def load(self, path, format):
        if format.lower() not in self.supported_formats:
            raise ValueError(
                f'Format "{format}" was not found in the list of supported data types.'
            )
        elif format.lower() == 'fasta':
            self._data = list(SeqIO.parse(path, "fasta"))
        
        elif format.lower() == 'gff3':
            pass

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

