def read_sequence(file_path):
    """
    Reads the sequence from a FASTA file, skipping headers and description lines.
    
    Args:
        file_path (str): The path to the FASTA file.
    
    Returns:
        str: The clean sequence string.
    """
    sequence = []
    skip_next = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                skip_next = True
                continue
            if skip_next:
                skip_next = False
                continue
            sequence.append(line)
    return ''.join(sequence)

def read_chars(num_chars, file_path="../shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files/Genome Reference -TAIR10--ChrM-1..367808.fasta", offset=0):
    """
    Reads the specified number of characters from the clean sequence starting from an offset.
    
    Args:
        num_chars (int): The number of characters to read.
        file_path (str): The path to the file to read from.
        offset (int): The offset in the clean sequence.
    
    Returns:
        str: The read characters.
    """
    full_seq = read_sequence(file_path)
    return full_seq[offset:offset + num_chars]