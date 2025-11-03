import random
from Bio import SeqIO


class GenomicDataLoader:
    def __init__(self):
        self.supported_formats = [
            'fasta',
            'gff3',
        ]

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

