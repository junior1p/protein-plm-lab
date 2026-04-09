#!/usr/bin/env python3
"""
Zero-shot mutation fitness evaluation using trained ESM2-small model.
Uses masked LM logit difference between WT and mutant sequences.
"""
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr


def compute_sequence_score(model, tokenizer, device, sequence):
    """
    Compute average MLM logit score for a sequence.
    Higher score = more "native-like" / high-fitness.
    """
    ids = tokenizer.encode(sequence)
    seq_len = len(ids)

    # Create masked input: mask a fraction of positions
    masked_ids = ids.copy()
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    mask_positions = [i for i, tok in enumerate(ids)
                      if tok not in (tokenizer.cls_token_id,
                                     tokenizer.sep_token_id,
                                     tokenizer.pad_token_id,
                                     tokenizer.unk_token_id)]

    if not mask_positions:
        return 0.0

    # Evaluate a few representative positions (avoid full forward per position)
    # For speed, mask all positions at once
    masked_ids = ids.copy()
    scores = []

    with torch.no_grad():
        for pos in mask_positions:
            masked = masked_ids.copy()
            masked[pos] = mask_id
            tokens = torch.tensor([masked], dtype=torch.long).to(device)

            logits = model(tokens)  # (1, seq_len, vocab_size)
            pred_token = logits[0, pos].argmax().item()

            # Score = logit of the original (true) token
            true_token = ids[pos]
            score = logits[0, pos, true_token].item()
            scores.append(score)

    return np.mean(scores)


# GFP test mutations with known phenotypes
GFP_MUTATIONS = [
    # (mutation, wt_aa, mut_aa, position, expected_effect)
    ("K7V",  "K", "V",  6,  "neutral"),
    ("K7I",  "K", "I",  6,  "neutral"),
    ("G66Y", "G", "Y", 65,  "brighter"),
    ("G66H", "G", "H", 65,  "dimer"),
]

GFP_WT_SEQ = (
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQ"
    "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILG"
    "HKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELT KDEVDRMTDELDGDVNGHKFSVR"
    "GEAGE DATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIF"
    "FKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNG"
    "NKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT"
    "TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFK"
    "EDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNG"
    "HKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYV"
    "QERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRD"
    "HLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPW"
    "PTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELK"
    "GIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELD"
    "GDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAM"
    "PEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDH"
    "AVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT"
    "GKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTL"
    "VNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDR"
    "MTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHD"
    "FFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNV"
    "FKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTL"
    "KFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKF"
    "EGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTK"
    "DEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDH"
    "MKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERD"
    "TNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGK"
    "LTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAE"
    "VKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNE"
    "LTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYP"
    "DHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFE"
    "RDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATY"
    "GKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTR"
    "AEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLL"
    "NELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSR"
    "YPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYN"
    "FERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDA"
    "TYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYK"
    "TRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQL"
    "LNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFS"
    "RYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLY"
    "NFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGD"
    "ATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNY"
    "KTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDR"
    "QLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQ"
    "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILG"
    "HKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSG"
    "EGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFK"
    "DDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNK"
    "VLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
    "LTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKE"
    "DGNILGHKLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHK"
    "FSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLYNFERDTNNVFKEIDHAVIDR"
)


def make_mutant(wt_seq, position, mut_aa):
    """Substitute amino acid at 0-indexed position."""
    seq_list = list(wt_seq)
    seq_list[position] = mut_aa
    return "".join(seq_list)


def evaluate_gfp(model, tokenizer, device):
    """Evaluate GFP mutations. Returns list of (mutation, type, wt_score, mut_score, delta)."""
    results = []
    for mut_name, wt_aa, mut_aa, pos, effect in GFP_MUTATIONS:
        try:
            # Truncate WT sequence for evaluation (use first 512 or full)
            wt_trunc = GFP_WT_SEQ[:512]
            mut_seq = make_mutant(wt_trunc, pos, mut_aa)

            wt_score = compute_sequence_score(model, tokenizer, device, wt_trunc)
            mut_score = compute_sequence_score(model, tokenizer, device, mut_seq)
            delta = mut_score - wt_score

            results.append((mut_name, effect, wt_score, mut_score, delta))
            print(f"  {mut_name} ({effect:9s}) | WT={wt_score:.2f} | Mut={mut_score:.2f} | Δ={delta:+.2f}")
        except Exception as e:
            print(f"  {mut_name}: ERROR - {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")  # cuda, cpu, mlu
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mlu") and torch.mlu.is_available():
            device = "mlu"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading model from {args.checkpoint} on {device}...")

    # Import from the repo
    import sys, os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    from train import ESM2Small, ProteinTokenizer

    tokenizer = ProteinTokenizer()
    model = ESM2Small(vocab_size=31, max_len=512)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("\n=== GFP Mutation Fitness Evaluation ===")
    results = evaluate_gfp(model, tokenizer, device)

    if len(results) >= 2:
        deltas = [r[4] for r in results]
        # Rank by effect: neutral=0, brighter=1, dimer=-1 (or negative for deleterious)
        # For Spearman, use reported ordering
        # Simple ranking: neutral=mid, bright/high=high, dimer=low
        rho, pval = spearmanr([0, 0, -1, -1], deltas)
        print(f"\n  Spearman ρ: {rho:.3f} (p={pval:.3f})")


if __name__ == "__main__":
    main()
