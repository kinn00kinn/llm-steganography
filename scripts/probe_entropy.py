"""Measure the entropy of Japanese text generation."""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from lsteg.model import ModelManifest, TransformersBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"

PROMPTS = [
    "今日は研究室で",
    "最近のAI技術の進歩について、",
    "大学の授業で学んだことの中で、",
    "週末は友達と",
    "美味しいコーヒーの淹れ方は",
    "日本の四季の魅力は",
    "将来の夢について",
    "最近読んだ本で面白かったのは",
    "健康のために気をつけていることは",
    "プログラミングの面白さは",
]

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--local-files-only", action="store_true")
    return parser

def sample_next_token(logits_list: list[float], temperature: float, top_p: float) -> tuple[int, float]:
    """Sample a token and return (token_id, entropy_in_bits)."""
    t_logits = torch.tensor(logits_list, dtype=torch.float32)
    if temperature != 1.0:
        t_logits = t_logits / temperature
        
    probs = F.softmax(t_logits, dim=-1)
    
    # Calculate entropy of the full distribution in bits
    entropy = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
    
    # top-p sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()
    
    token_id = torch.multinomial(probs, 1).item()
    return token_id, entropy

def main():
    args = build_parser().parse_args()
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(
        manifest, cache_dir=args.cache_dir, local_files_only=args.local_files_only
    )
    
    entropies = []
    total_chars = 0
    total_tokens = 0
    
    for i in range(args.samples):
        prompt = random.choice(PROMPTS)
        token_ids = backend.tokenize(prompt)
        
        generated_tokens = []
        for _ in range(args.max_new_tokens):
            logits = backend.next_logits(token_ids)
            token_id, entropy = sample_next_token(list(logits), args.temperature, args.top_p)
            
            token_ids.append(token_id)
            generated_tokens.append(token_id)
            entropies.append(entropy)
            
            # Break on Qwen EOS tokens (151643 is <|endoftext|>, 151645 is <|im_end|>)
            if token_id in (151643, 151645):
                break
                
        text = backend.detokenize(generated_tokens)
        total_chars += len(text)
        total_tokens += len(generated_tokens)
        
        print(f"Sample {i+1:3d}: generated {len(text):3d} chars, {len(generated_tokens):3d} tokens", flush=True)

    if not entropies:
        print("No tokens generated.")
        return 0
        
    mean_entropy = statistics.mean(entropies)
    median_entropy = statistics.median(entropies)
    sorted_entropies = sorted(entropies)
    p10 = sorted_entropies[int(len(sorted_entropies)*0.1)]
    p90 = sorted_entropies[int(len(sorted_entropies)*0.9)]
    tokens_per_char = total_tokens / total_chars if total_chars else 0
    
    tokens_for_500_chars = 500 * tokens_per_char
    theoretical_payload = tokens_for_500_chars * mean_entropy
    
    print("\n--- Results ---")
    print(f"mean entropy     = {mean_entropy:.2f} bits/token")
    print(f"median entropy   = {median_entropy:.2f}")
    print(f"p10              = {p10:.2f}")
    print(f"p90              = {p90:.2f}")
    print(f"\ntoken / character = {tokens_per_char:.3f}")
    print(f"\n500 Unicode chars ≈ {tokens_for_500_chars:.1f} tokens")
    print(f"理論payload (500 chars) ≈ {theoretical_payload:.1f} bits")
    
    # 100 characters is typically ~200-300 bytes when compressed.
    # Plus AEAD overhead (~50 bytes) -> ~250-350 bytes -> 2000-2800 bits.
    print(f"\nGO/NO-GO 判定基準: 理論payloadが 2500〜3000 bits 程度を超えているか？")
    return 0

if __name__ == "__main__":
    sys.exit(main())
