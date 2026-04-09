---
name: train-esm2-small
description: End-to-end training of ESM2-small (9.6M-parameter protein language model) on Swiss-Prot — data download, tokenization, training, checkpointing, evaluation, and model upload to GitHub. Works on GPU (CUDA) and Cambricon MLU370.
version: 1.0.0
author: Max
license: MIT
dependencies: [torch>=2.0, tqdm, requests]
metadata:
  hermes:
    tags: [protein language model, ESM-2, masked language modeling, MLU370, protein engineering, PyTorch, MLM]
    repo: https://github.com/junior1p/ESM2-small
---

# Train ESM2-small: Protein Language Model from Scratch

Train a compact 9.6M-parameter ESM-2 architecture on Swiss-Prot protein sequences end-to-end.

## When to Use This Skill

- Training a protein language model from scratch
- Evaluating zero-shot mutation prediction (fitness)
- Adapting the ESM-2 architecture to new protein datasets
- Setting up checkpointing and resume for long training runs
- Uploading trained models to GitHub Release

## Quick Start

```bash
# 1. Download data
python scripts/download_data.py

# 2. Train (GPU)
python train.py --data data/swissprot_train.fasta --val_data data/swissprot_val.fasta \
  --out_dir output --epochs 5 --batch_size 32 --device cuda

# 3. Evaluate zero-shot fitness
python scripts/evaluate_fitness.py --checkpoint output/checkpoint_final_best.pt
```

## Training Pipeline

### Data Download

```bash
python scripts/download_data.py
```

Downloads Swiss-Prot curated protein sequences from UniProt, splits into train/val, and saves FASTA files.

- Output: `data/swissprot_train.fasta`, `data/swissprot_val.fasta`
- Train: 433,583 sequences | Val: 22,821 sequences
- Truncation: sequences > 512 tokens are truncated at C-terminus

### Tokenizer

31-token amino acid vocabulary:
- 20 standard amino acids (A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V)
- Special: `[MASK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[UNK]`

### Architecture

ESM2-small mirrors ESM-2 "mini" (12-layer Transformer):

| Parameter | Value |
|---|---|
| Layers | 12 |
| Hidden dim | 256 |
| Attention heads | 8 |
| FFN dim | 1024 |
| Vocab size | 31 |
| Max length | 512 |
| Total params | **9,624,607** |

### Training Configuration

| Parameter | Default |
|---|---|
| Optimizer | AdamW (lr=1e-4, β=(0.9, 0.999), eps=1e-8, weight_decay=0.01) |
| LR schedule | Linear warmup 1000 steps → cosine decay to ~1e-9 |
| Batch size | 32 sequences |
| Masking | 15% uniform random |
| Mixed precision | FP32 weights, BF16 forward/backward (MLU370) |
| Epochs | 5 (~2h/epoch on single MLU370, ~1h/epoch on A100) |
| Steps per epoch | ~13,500 |
| Throughput | ~30K tokens/sec (single card) |
| Total tokens | ~1.1 billion |

### Checkpointing

Three checkpoint types are saved automatically:

1. **Periodic** (every 2000 steps): full snapshot for resume
2. **Epoch** (end of each epoch): history checkpoint
3. **Best** (when val loss improves): `*_best.pt` copy

Checkpoint contents:
```python
{
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict,       # model weights
    "optimizer_state_dict": dict,    # AdamW state
    "lr_scheduler_state_dict": dict, # cosine schedule position
    "rng_state": dict,               # torch/cuda/numpy RNG for reproducibility
    "train_loss": float,
    "val_loss": float,
    "config": dict,                  # hyperparameters
}
```

### Resume from Checkpoint

```bash
python train.py --resume output/checkpoint_step20000.pt --data ...
```

Resumes exact training state (model weights + optimizer + LR schedule + RNG seed).

## Evaluation: Zero-Shot Fitness Prediction

```bash
python scripts/evaluate_fitness.py --checkpoint output/checkpoint_final_best.pt
```

Uses masked language modeling logit difference for zero-shot variant effect prediction:

1. Encode wild-type (WT) sequence → get per-position MLM logits
2. Encode mutant sequence → get per-position MLM logits
3. $\Delta = \text{Score}(\text{mutant}) - \text{Score}(\text{WT})$

Positive $\Delta$ → potentially beneficial mutation
Negative $\Delta$ → potentially deleterious mutation

Tested on GFP (Green Fluorescent Protein) mutations:

| Mutation | Type | Expected Δ |
|---|---|---|
| K7V | neutral | ~0 |
| K7I | neutral | negative |
| G66Y | brighter | large negative (unexpected direction) |
| G66H | dimer | large negative |

## Scripts

### `scripts/download_data.py`

Downloads Swiss-Prot from UniProt REST API, deduplicates, splits 95/5 train/val.

```bash
python scripts/download_data.py
# Options:
#   --output_dir ./data     # default: ./data
#   --split_ratio 0.95     # default: 0.95
#   --max_len 512          # default: 512
```

### `scripts/evaluate_fitness.py`

Zero-shot mutation prediction using trained model.

```bash
python scripts/evaluate_fitness.py \
    --checkpoint output/checkpoint_final_best.pt \
    --device cuda
# Options:
#   --device cuda/cpu/mlu  # default: auto-detect
```

## Model Upload to GitHub

After training, upload model weights to GitHub Release:

```bash
# Using GitHub CLI
gh release create v1.0.0 \
    --title "ESM2-small v1.0.0" \
    --notes "Trained on Swiss-Prot, val_loss=0.417"

gh release upload v1.0.0 output/checkpoint_final_best.pt

# Or using the upload script
python scripts/upload_to_github.py \
    --token hf_xxxx \
    --repo junior1p/ESM2-small \
    --tag v1.0.0
```

## Output Files

```
output/
├── config.json                    # training config
├── checkpoint_epoch1.pt           # epoch snapshots
├── checkpoint_epoch1_best.pt
├── checkpoint_epoch2.pt
├── ...
├── checkpoint_final.pt            # final epoch
├── checkpoint_final_best.pt       # best val loss model ← USE THIS
├── checkpoint_step2000.pt         # periodic snapshots (every 2000 steps)
├── checkpoint_step4000.pt
└── ...
```

## Training from Existing Code

The full training pipeline is in `train.py` at the repo root. Key entry point:

```python
from train import ESM2Small, ProteinTokenizer, train_model

model = ESM2Small(vocab_size=31, max_len=512)
tokenizer = ProteinTokenizer()

train_model(
    model=model,
    tokenizer=tokenizer,
    train_data="data/swissprot_train.fasta",
    val_data="data/swissprot_val.fasta",
    out_dir="output",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    device="cuda",
)
```

## Known Limitations

- GFP Spearman ρ ≈ 0.200 (small model, short training — larger models achieve 0.3–0.5)
- No MSA or structure features — pure MLM only
- Single card training — multi-card scaling not included
- No dropout — model is meant for transfer/fine-tuning, not direct deployment without adaptation

## Pitfalls

- **Resume requires `--resume` flag AND re-pass `--data`**: The data loader is not saved in checkpoints
- **Truncated sequences**: Sequences > 512 tokens are cut at C-terminus — may lose C-terminal functional domains
- **Val loss > Train loss**: Normal for protein MLM — val sequences are unseen proteins, not random noise
- **MLU370 BF16**: If training on MLU370, ensure CNNL operators support BF16 forward pass; fallback to FP32 if needed
- **Checkpoint disk space**: Each checkpoint is ~115MB (FP32 model + optimizer). Budget ~5GB for a full training run with 2000-step periodic saves.

## Reproducibility

```bash
git clone https://github.com/junior1p/ESM2-small.git
cd ESM2-small
python scripts/download_data.py
python train.py --data data/swissprot_train.fasta --val_data data/swissprot_val.fasta \
    --out_dir output --epochs 5 --seed 42
```
