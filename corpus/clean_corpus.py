import re, pathlib, unicodedata, sys, textwrap
root = pathlib.Path('.')
START_RE = re.compile(r'\*\*\*\s*START OF [^\n]+\n', re.I)
END_RE   = re.compile(r'\*\*\*\s*END OF [^\n]+\n', re.I)
FOOTNOTE = re.compile(r'\[\d+\]|\(\d+\)|\{\d+\}|†')

def clean(text: str) -> str:
    # 1. strip boiler-plate
    text = START_RE.split(text, 1)[-1]       # after START
    text = END_RE.split(text, 1)[0]          # before END

    # 2. unicode normalise & ascii fallback
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 3. remove footnote markers
    text = FOOTNOTE.sub('', text)
    
    # 4. collapse whitespace
    text = re.sub(r'\r\n?', '\n', text)      # CRLF → LF
    text = re.sub(r'\n{3,}', '\n\n', text)   # >2 blanks → 1
    text = re.sub(r'[ \t]{2,}', ' ', text)   # runs of spaces
    return text.strip()

out_lines = []
for fp in root.glob('pg*.txt'):
    txt = clean(fp.read_text(encoding='utf-8', errors='ignore'))
    seen = set()
    # 5. naive line-dedupe (good enough)
    for line in txt.splitlines():
        if line and line not in seen:
            seen.add(line)
            out_lines.append(line)

out = '\n'.join(out_lines)
pathlib.Path('corpus_clean.txt').write_text(out)
print(f"✅ wrote corpus_clean.txt ({len(out)/1e6:.2f} MB)")
