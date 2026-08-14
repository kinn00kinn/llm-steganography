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
    "架空のニュース記事：\n本日午後、東京都内の",
    "【独自速報】\n新しい再生可能エネルギーの技術が",
    "国際経済の展望：\n来年度の市場動向について、",
    "文化庁発表：\n全国の伝統工芸品に関する新たな助成制度が",
    "スポーツ特集：\n昨日行われた決勝戦で、両チームは",
    "テクノロジー動向：\n次世代の量子コンピューター開発において、",
    "深海探査の最新レポート：\n太平洋の海底火山付近で、",
    "都市開発プロジェクト：\n首都圏における新しい公共交通機関の",
    "医療の最前線：\n最新の遺伝子治療に関する臨床試験が",
    "歴史発見：\n奈良県の遺跡から、これまで知られていなかった",
]

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    # For Phase 5A: explicitly indicate thinking mode is disabled in generation
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode (default False)")
    return parser

def sample_next_token(logits_list: list[float], temperature: float, top_p: float, top_k: int) -> tuple[int, float]:
    """Sample a token and return (token_id, entropy_in_bits)."""
    t_logits = torch.tensor(logits_list, dtype=torch.float32)
    if temperature != 1.0:
        t_logits = t_logits / temperature
        
    probs = F.softmax(t_logits, dim=-1)
    
    # Calculate entropy of the full distribution in bits
    entropy = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
    
    # top-k sampling
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        probs_new = torch.zeros_like(probs)
        probs_new.scatter_(0, top_k_indices, top_k_probs)
        probs = probs_new / probs_new.sum()
        
    # top-p sampling
    if top_p < 1.0:
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
    
    capacity_per_500_chars = []
    
    for i in range(args.samples):
        prompt = random.choice(PROMPTS)
        token_ids = backend.tokenize(prompt)
        
        generated_tokens = []
        seq_entropy = 0.0
        
        for _ in range(args.max_new_tokens):
            logits = backend.next_logits(token_ids)
            token_id, entropy = sample_next_token(list(logits), args.temperature, args.top_p, args.top_k)
            
            token_ids.append(token_id)
            generated_tokens.append(token_id)
            seq_entropy += entropy
            
            # Break on Qwen EOS tokens (151643 is <|endoftext|>, 151645 is <|im_end|>)
            if token_id in (151643, 151645):
                break
                
        text = backend.detokenize(generated_tokens)
        chars = len(text)
        
        if chars > 0:
            bits_per_char = seq_entropy / chars
            cap_500 = bits_per_char * 500
            capacity_per_500_chars.append(cap_500)
        
        print(f"Sample {i+1:3d}: generated {chars:3d} chars, {len(generated_tokens):3d} tokens, capacity for 500 chars: {cap_500:.1f} bits", flush=True)

    if not capacity_per_500_chars:
        print("No tokens generated.")
        return 0
        
    def get_p(arr, p):
        return sorted(arr)[int(len(arr)*p)]
        
    mean_cap = statistics.mean(capacity_per_500_chars)
    p05 = get_p(capacity_per_500_chars, 0.05)
    p10 = get_p(capacity_per_500_chars, 0.10)
    p25 = get_p(capacity_per_500_chars, 0.25)
    p50 = get_p(capacity_per_500_chars, 0.50)
    p75 = get_p(capacity_per_500_chars, 0.75)
    p90 = get_p(capacity_per_500_chars, 0.90)
    
    print("\n--- Capacity for 500 characters ---")
    print(f"mean   : {mean_cap:.1f} bits")
    print(f"p05    : {p05:.1f} bits")
    print(f"p10    : {p10:.1f} bits")
    print(f"p25    : {p25:.1f} bits")
    print(f"median : {p50:.1f} bits")
    print(f"p75    : {p75:.1f} bits")
    print(f"p90    : {p90:.1f} bits")
    
    # 500-character goal requires enough capacity for secret+crypto
    # Currently expected: ~350 bit secret + ~150 bit crypto = ~500 bits
    print(f"\nPhase 5D 判定基準目安: p10 が約 500 bits を超えているか？")
    return 0

if __name__ == "__main__":
    sys.exit(main())
