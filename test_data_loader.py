"""
Test script to verify genomic_data_loader behavior and inspect sequences.
Checks for N content and prints sample sequences from TAIR10.
"""

from genomic_data_loader import GenomicDataLoader
from omegaconf import OmegaConf

try:
    from evo2 import Evo2
    EVO2_AVAILABLE = True
except ImportError:
    EVO2_AVAILABLE = False
    print("Warning: Evo2 not available, skipping tokenizer vocabulary check")

def analyze_n_content(seq_str):
    """Analyze N content in a sequence."""
    n_count = seq_str.upper().count('N')
    total = len(seq_str)
    pct = (n_count / total) * 100 if total > 0 else 0
    return n_count, pct

def check_evo2_vocabulary():
    """Check if Evo2 tokenizer includes 'N' in its vocabulary."""
    if not EVO2_AVAILABLE:
        print("\n  Evo2 not available - skipping vocabulary check")
        return
    
    print("\n" + "=" * 80)
    print("TEST 4: EVO2 TOKENIZER VOCABULARY CHECK")
    print("=" * 80)
    
    try:
        # Load a small model to check tokenizer (don't need full model)
        print("\nLoading Evo2 model to access tokenizer...")
        model = Evo2('evo2_1b_base')
        tokenizer = model.tokenizer
        
        # Get vocabulary
        print(f"\nTokenizer type: {type(tokenizer).__name__}")
        
        # Try to access vocabulary
        if hasattr(tokenizer, 'vocab'):
            vocab = tokenizer.vocab
            print(f"Vocabulary size: {len(vocab)}")
            print(f"\nFull vocabulary: {vocab}")
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            print(f"Vocabulary size: {len(vocab)}")
            print(f"\nFull vocabulary: {vocab}")
        elif hasattr(tokenizer, 'char_to_id'):
            # Character-level tokenizer
            print("\nCharacter-level tokenizer detected")
            vocab = tokenizer.char_to_id if hasattr(tokenizer, 'char_to_id') else {}
            if vocab:
                print(f"Vocabulary size: {len(vocab)}")
                print(f"\nFull vocabulary: {sorted(vocab.keys())}")
        else:
            # Try tokenizing to see what happens
            print("\nDirect vocabulary access not available, testing tokenization...")
            test_chars = ['A', 'T', 'C', 'G', 'N', 'n']
            print("\nTesting individual characters:")
            for char in test_chars:
                try:
                    tokens = tokenizer.tokenize(char)
                    print(f"  '{char}' -> tokens: {tokens}")
                except Exception as e:
                    print(f"  '{char}' -> ERROR: {e}")
        
        # Check specifically for N
        print("\n" + "-" * 80)
        print("CHECKING FOR 'N' IN VOCABULARY:")
        print("-" * 80)
        
        has_uppercase_n = False
        has_lowercase_n = False
        
        if hasattr(tokenizer, 'vocab'):
            vocab = tokenizer.vocab
            has_uppercase_n = 'N' in vocab
            has_lowercase_n = 'n' in vocab
        elif hasattr(tokenizer, 'char_to_id'):
            vocab = tokenizer.char_to_id
            has_uppercase_n = 'N' in vocab
            has_lowercase_n = 'n' in vocab
        else:
            # Test by tokenizing
            try:
                n_tokens = tokenizer.tokenize('N')
                has_uppercase_n = len(n_tokens) > 0
                n_lower_tokens = tokenizer.tokenize('n')
                has_lowercase_n = len(n_lower_tokens) > 0
            except:
                pass
        
        print(f"\n✓ Contains 'N' (uppercase): {has_uppercase_n}")
        print(f"✓ Contains 'n' (lowercase): {has_lowercase_n}")
        
        if has_uppercase_n or has_lowercase_n:
            print("\n SUCCESS: 'N' is in the vocabulary - model can handle ambiguous bases!")
        else:
            print("\n  WARNING: 'N' not found in vocabulary - model may not handle ambiguous bases properly!")
        
        # Test tokenizing an N-rich sequence
        print("\n" + "-" * 80)
        print("TESTING N-RICH SEQUENCE TOKENIZATION:")
        print("-" * 80)
        test_seq = "ATCGNNNNATCG"
        print(f"\nTest sequence: {test_seq}")
        try:
            tokens = tokenizer.tokenize(test_seq)
            print(f"Tokenized: {tokens}")
            print(f"Number of tokens: {len(tokens)}")
        except Exception as e:
            print(f"ERROR tokenizing: {e}")
        
    except Exception as e:
        print(f"\n Error checking vocabulary: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Load config
    config = OmegaConf.load('configs/eval/custom_eval_1.yaml')
    run = list(config.runs.values())[0]
    
    data_path = run.data.path
    data_format = run.data.format
    
    print("=" * 80)
    print("GENOMIC DATA LOADER TEST")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Format: {data_format}\n")
    
    # Initialize loader
    loader = GenomicDataLoader()
    loader.load(path=data_path, format=data_format, verbose=True)
    
    # Test 1: Print first 100 bases of each chromosome
    print("\n" + "=" * 80)
    print("TEST 1: First 100 bases of each sequence")
    print("=" * 80)
    
    for idx, seq in enumerate(loader._data, 1):
        first_100 = str(seq.seq[:100])
        n_count, n_pct = analyze_n_content(first_100)
        
        print(f"\nSequence {idx}: {seq.id}")
        print(f"  First 100 bases: {first_100}")
        print(f"  N content: {n_count} Ns ({n_pct:.1f}%)")
    
    # Test 2: Test read_start with different sample sizes
    print("\n" + "=" * 80)
    print("TEST 2: Testing read_start() with seq_len=32, pred_len=128")
    print("=" * 80)
    
    # Initialize 4 unique samples
    num_samples = min(4, len(loader._data))
    loader.initialize_unique_samples(num_samples=num_samples)
    
    print(f"\nSelected {num_samples} unique samples for testing:")
    for idx, seq in enumerate(loader._selected_samples, 1):
        print(f"  Sample {idx}: {seq.id} (length: {len(seq):,})")
    
    print("\nReading 8 samples (cycling through the selected sequences):")
    for i in range(8):
        input_seq, label_seq = loader.read_start(splits=[32, 128])
        
        input_n_count, input_n_pct = analyze_n_content(input_seq)
        label_n_count, label_n_pct = analyze_n_content(label_seq)
        
        print(f"\n  Read {i+1}:")
        print(f"    Input (32 bases): {input_seq}")
        print(f"    Input N content: {input_n_count} Ns ({input_n_pct:.1f}%)")
        print(f"    Label (128 bases): {label_seq[:50]}...")
        print(f"    Label N content: {label_n_count} Ns ({label_n_pct:.1f}%)")
    
    # Test 3: Overall N content statistics
    print("\n" + "=" * 80)
    print("TEST 3: Overall N Content Statistics")
    print("=" * 80)
    
    for idx, seq in enumerate(loader._data, 1):
        full_seq = str(seq.seq)
        n_count, n_pct = analyze_n_content(full_seq)
        
        # Also check first 1000 bases
        first_1k = full_seq[:1000]
        n_count_1k, n_pct_1k = analyze_n_content(first_1k)
        
        print(f"\n{seq.id}:")
        print(f"  Total length: {len(full_seq):,} bases")
        print(f"  Total N content: {n_count:,} Ns ({n_pct:.2f}%)")
        print(f"  First 1000 bases N content: {n_count_1k} Ns ({n_pct_1k:.1f}%)")
    
    # Test 4: Check Evo2 vocabulary
    check_evo2_vocabulary()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

