
import sys
import csv

def convert_gtf_to_gff3(gtf_path, gff3_path):
    print(f"Converting {gtf_path} to {gff3_path}...")
    with open(gtf_path, 'r') as f_in, open(gff3_path, 'w') as f_out:
        f_out.write("##gff-version 3\n")
        
        for line in f_in:
            if line.startswith("#"):
                f_out.write(line)
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
                
            # Filter invalid strands
            if parts[6] not in ['+', '-']:
                continue
                
            attributes_str = parts[8]
            # Parse GTF attributes: key "value";
            attr_map = {}
            for chunk in attributes_str.split(';'):
                chunk = chunk.strip()
                if not chunk: continue
                if ' "' in chunk:
                    key, val = chunk.split(' "', 1)
                    val = val[:-1] # remove trailing "
                elif ' ' in chunk:
                    key, val = chunk.split(' ', 1)
                else:
                    key = chunk
                    val = "" # Flag?
                attr_map[key] = val
            
            # Construct GFF3 attributes
            new_attrs = []
            
            # ID and Parent logic matching typical NCBI GTF to GFF3 mapping
            # This is heuristic.
            
            feat_type = parts[2]
            
            if feat_type == "gene":
                if "gene_id" in attr_map:
                    new_attrs.append(f"ID={attr_map['gene_id']}")
                    if "gene" in attr_map: new_attrs.append(f"Name={attr_map['gene']}")
            elif feat_type in ["mRNA", "transcript", "tRNA", "rRNA"]:
                 if "transcript_id" in attr_map:
                    new_attrs.append(f"ID={attr_map['transcript_id']}")
                    if "gene_id" in attr_map:
                        new_attrs.append(f"Parent={attr_map['gene_id']}")
            elif feat_type in ["exon", "CDS", "UTR"]:
                 # Exons don't always have IDs in GTF, but need Parent
                 if "transcript_id" in attr_map:
                    new_attrs.append(f"Parent={attr_map['transcript_id']}")
                 elif "gene_id" in attr_map:
                    new_attrs.append(f"Parent={attr_map['gene_id']}")
            
            # Copy other interesting attributes
            for k, v in attr_map.items():
                if k not in ["gene_id", "transcript_id"]:
                     new_attrs.append(f"{k}={v}")
            
            parts[8] = ";".join(new_attrs)
            f_out.write("\t".join(parts) + "\n")
            
    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gtf2gff3.py <input.gtf> <output.gff3>")
        sys.exit(1)
    convert_gtf_to_gff3(sys.argv[1], sys.argv[2])
