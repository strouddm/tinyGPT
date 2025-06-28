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
python generate.py --prompt "Who is the son of God?"
```

Example Results:

```bash
Starting TinyGPT Text Generation...
📖 Prompt: 'Who is the son of God'
🎯 Target tokens: 100
🌡️  Temperature: 0.7
🔧 Loading model and tokenizer...
   Vocabulary size: 50,257
📂 Loading checkpoint: checkpoint_epoch5.pt
✅ Model loaded and ready for generation

============================================================
🎭 Starting generation...
   Initial prompt tokens: 6
   Generated 10/100 tokens...
   Generated 20/100 tokens...
   Generated 30/100 tokens...
   Generated 40/100 tokens...
   Generated 50/100 tokens...
   Generated 60/100 tokens...
   Generated 70/100 tokens...
   Generated 80/100 tokens...
   Generated 90/100 tokens...
✅ Generation complete! Total tokens: 106
============================================================
🎉 GENERATED TEXT:
============================================================
Who is the son of God, and whom God has given. For
he says, "Hine own son shall send forth Thy law;" and
in which seditions, "Preareth shall offer up," and what
dantec to the judgment of choice, and to "His
commandment." He adds, "The Lord hath rent up the people of God,"
otherwise than "His grace," and "His coming up the Lord Jesus,"
He adds, "And it shall not be greater than
============================================================
```


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
