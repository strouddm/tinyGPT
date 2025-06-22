# tiny_gpt_cpu
TinyGPT that trains a ~42 M‑parameter language model on a small custom corpus of western church philosophers from Gutenberg: Augustine and Saint Thomas Aquinas

## Quick start

```bash
# clone the repo (or unzip the archive you downloaded)
cd tiny_gpt_cpu
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

1. Place your cleaned corpus at `corpus/corpus_clean.txt`.
   *Use `clean_corpus.py` if you’re using Project Gutenberg sources.*

2. Train:

```bash
python train.py
```

3. Generate text from the trained checkpoint:

```bash
python generate.py --prompt "Who is God?"
```

The first run on an 8‑core laptop should take ~4–6 h for 5 epochs on a 15 MB corpus.

## Directory layout
```
tiny_gpt_cpu/
├── train.py              # main training entry‑point
├── generate.py           # sampling helper
├── model.py              # TinyGPT architecture
├── requirements.txt
└── corpus/
    ├── clean_corpus.py   # Clean downloaded gutenberg files in pg[number].txt format
    └── corpus_clean.txt  # Cleaned corpus of text
```

## Requirements
* Python ≥3.9
* PyTorch ≥2.2.1 (CPU build is fine)
* numpy, tiktoken