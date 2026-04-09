#!/usr/bin/env python3
"""
Download Swiss-Prot protein sequences from UniProt, deduplicate,
and split into train/validation sets.
"""
import argparse
import os
import random
import time
import requests
from collections import defaultdict


def download_swissprot_fasta(output_dir="./data", chunk_size=100000):
    """Download Swiss-Prot FASTA from UniProt REST API."""
    os.makedirs(output_dir, exist_ok=True)
    fasta_path = os.path.join(output_dir, "swissprot_raw.fasta")

    if os.path.exists(fasta_path):
        print(f"Using cached: {fasta_path}")
        return fasta_path

    url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=*&fields=accession,sequence&size=500"
    print(f"Downloading Swiss-Prot from UniProt...")

    sequences = []
    offset = 0
    session = requests.Session()

    while True:
        resp = session.get(f"{url}&offset={offset}", timeout=60)
        resp.raise_for_status()
        chunk = resp.text

        if len(chunk) < 10 or "INTERNAL SERVER ERROR" in chunk:
            print(f"  Offset {offset}: got error, retrying...")
            time.sleep(5)
            continue

        sequences.append(chunk)
        count = chunk.count("\n>")
        print(f"  Downloaded {offset + count:,} sequences...", end="\r")

        if count < 500 or "\n" not in chunk:
            break
        offset += 500
        time.sleep(0.3)  # be nice to UniProt API

    print(f"\nTotal downloaded: {sum(s.count(chr(10)+'>') for s in sequences):,} entries")

    full = "".join(sequences)
    with open(fasta_path, "w") as f:
        f.write(full)

    print(f"Saved to {fasta_path} ({len(full)//1024//1024} MB)")
    return fasta_path


def parse_fasta(path):
    """Yield (header, sequence) from FASTA file."""
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def deduplicate_and_split(fasta_path, output_dir, split_ratio=0.95, max_len=512,
                          min_len=10, seed=42):
    """
    Parse FASTA, deduplicate by sequence, filter by length,
    split into train/val by sequence identity.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Deduplicate by sequence
    print("Deduplicating...")
    seq_map = defaultdict(list)  # seq -> list of headers
    for header, seq in parse_fasta(fasta_path):
        seq = seq.upper().replace(" ", "")
        if min_len <= len(seq) <= max_len:
            seq_map[seq].append(header)

    unique_seqs = list(seq_map.keys())
    random.shuffle(unique_seqs)
    print(f"Unique sequences: {len(unique_seqs):,} (from {sum(len(v) for v in seq_map.values()):,} total)")

    # Split
    n_train = int(len(unique_seqs) * split_ratio)
    train_seqs = unique_seqs[:n_train]
    val_seqs = unique_seqs[n_train:]

    def write_fasta(seqs, out_path):
        with open(out_path, "w") as f:
            for seq in seqs:
                headers = seq_map[seq]
                # Use first header as representative
                repr_header = headers[0]
                # Add duplicate count to header
                if len(headers) > 1:
                    repr_header = f"{repr_header} ndup={len(headers)}"
                f.write(f">{repr_header}\n")
                # Write sequence in 60-char lines
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + "\n")

    train_path = os.path.join(output_dir, "swissprot_train.fasta")
    val_path = os.path.join(output_dir, "swissprot_val.fasta")

    write_fasta(train_seqs, train_path)
    write_fasta(val_seqs, val_path)

    print(f"Train: {len(train_seqs):,} sequences -> {train_path}")
    print(f"Val:   {len(val_seqs):,} sequences -> {val_path}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Download and split Swiss-Prot")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--split_ratio", type=float, default=0.95)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fasta_path = download_swissprot_fasta(args.output_dir)
    train_path, val_path = deduplicate_and_split(
        fasta_path, args.output_dir,
        split_ratio=args.split_ratio,
        max_len=args.max_len,
        min_len=args.min_len,
        seed=args.seed
    )
    print("\nDone! Train on:")
    print(f"  python train.py --data {train_path} --val_data {val_path} ...")


if __name__ == "__main__":
    main()
