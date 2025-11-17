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

    def load(self, path, format, verbose=False):
        if format.lower() not in self.supported_formats:
            raise ValueError(
                f'Format "{format}" was not found in the list of supported data types.'
            )
        elif format.lower() == 'fasta':
            self._data = list(SeqIO.parse(path, "fasta"))
            
            # Print information about loaded sequences
            if verbose:
                print(f"\nLoaded {len(self._data)} sequences from {path}")
                print("=" * 70)
                for idx, seq in enumerate(self._data, 1):
                    seq_len = len(seq)
                    mid_point = seq_len // 2
                    print(f"  Sample {idx}: {seq.id[:40]:40} | Length: {seq_len:8,} | Midpoint: {mid_point:8,}")
                print("=" * 70)
                # Calculate average sequence length
                total_length = 0
                count = 0
                for seq in self._data:
                    total_length += len(seq)
                    count += 1
                avg_length = total_length / count if count > 0 else 0
                print(f"Average sequence length: {avg_length:,.0f}")
                print()
        
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

