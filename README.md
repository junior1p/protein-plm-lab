# Protein PLM Lab

End-to-end training kit for compact protein language models.

## What's Inside

- **ESM2-small**: 9.6M-parameter protein language model trained on Swiss-Prot (val loss=0.417)
- **Training skill**: Complete Hermes skill for training PLMs from scratch
- **Data pipeline**: Swiss-Prot download, deduplication, train/val split
- **Evaluation**: Zero-shot mutation fitness prediction
- **Papers**: Model paper + Training systems paper (author: Max)

## Quick Start

```bash
# Clone
git clone https://github.com/junior1p/protein-plm-lab.git
cd protein-plm-lab

# Download data
python scripts/download_data.py

# Train
python train.py --data data/swissprot_train.fasta --val_data data/swissprot_val.fasta \
    --out_dir output --epochs 5

# Evaluate fitness
python scripts/evaluate_fitness.py --checkpoint output/checkpoint_final_best.pt
```

## Papers

See `papers/` directory:
- `ESM2-small_model_paper.pdf` — Model architecture and zero-shot evaluation
- `ESM2-small_training_paper.pdf` — Training systems and MLU370 implementation

## Skill

The Hermes skill `skills/train-esm2-small/` contains full documentation for reproducing the training pipeline.

## License

MIT
