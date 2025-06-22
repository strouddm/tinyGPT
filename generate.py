import argparse, torch, torch.nn.functional as F, tiktoken
from model import TinyGPT

BLOCK_SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoint_epoch5.pt')
parser.add_argument('--prompt', required=True)
parser.add_argument('--tokens', type=int, default=100)
parser.add_argument('--temperature', type=float, default=0.7)
args = parser.parse_args()

print("🚀 Starting TinyGPT Text Generation...")
print(f"📖 Prompt: '{args.prompt}'")
print(f"🎯 Target tokens: {args.tokens}")
print(f"🌡️  Temperature: {args.temperature}")

print("🔧 Loading model and tokenizer...")
tokenizer = tiktoken.get_encoding('gpt2')
model = TinyGPT(tokenizer.n_vocab, BLOCK_SIZE)
print(f"   Vocabulary size: {tokenizer.n_vocab:,}")

print(f"📂 Loading checkpoint: {args.checkpoint}")
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
model.eval()
print("✅ Model loaded and ready for generation")

def generate(prompt, max_tokens, temp):
    print(f"🎭 Starting generation...")
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, :]
    print(f"   Initial prompt tokens: {len(idx[0])}")
    
    generated_text = prompt
    for token_idx in range(max_tokens):
        # Show progress every 10 tokens
        if token_idx % 10 == 0 and token_idx > 0:
            print(f"   Generated {token_idx}/{max_tokens} tokens...")
        
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    
    print(f"✅ Generation complete! Total tokens: {len(idx[0])}")
    return tokenizer.decode(idx[0].tolist())

print("\n" + "="*60)
result = generate(args.prompt, args.tokens, args.temperature)
print("="*60)
print("🎉 GENERATED TEXT:")
print("="*60)
print(result)
print("="*60)