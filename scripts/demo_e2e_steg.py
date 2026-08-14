"""End-to-End Steganography Spike using Cover LLM Range Coding."""

import argparse
import sys
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from lsteg.coding.frequencies import FrequencyTable
from lsteg.coding.range_coder import CodedBits, RangeDecoder, RangeEncoder
from lsteg.model import ModelManifest, TransformersBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"


def get_frequencies(
    logits_list: list[float], top_k: int = 256, total: int = 32768
) -> tuple[list[int], FrequencyTable]:
    """Convert logits to a FrequencyTable of exact size `total` using top_k tokens."""
    t_logits = torch.tensor(logits_list, dtype=torch.float32)
    probs = F.softmax(t_logits, dim=-1)

    # Get top-k probabilities and indices
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    top_k_probs = top_k_probs / top_k_probs.sum()

    # Assign minimum frequency of 1 to all top-k tokens
    freqs = torch.ones(top_k, dtype=torch.int32)
    remaining_total = total - top_k

    # Proportional allocation
    alloc = (top_k_probs * remaining_total).to(torch.int32)
    freqs += alloc

    # Fix rounding errors by adding remainder to the highest probability token
    remainder = total - freqs.sum().item()
    freqs[0] += remainder

    # Ensure no zero frequencies (already handled by ones) and total is correct
    assert freqs.sum().item() == total

    return top_k_indices.tolist(), FrequencyTable(freqs.tolist())


def hide_bits(backend: TransformersBackend, prompt: str, bits: CodedBits) -> list[int]:
    """Hide bits in generated text (Sender)."""
    token_ids = backend.tokenize(prompt)
    decoder = RangeDecoder(bits)
    generated = []

    print(f"Hiding {bits.bit_length} bits...")

    while True:
        if decoder._reader._position > bits.bit_length + 32:
            break

        try:
            logits = backend.next_logits(token_ids)
            indices, table = get_frequencies(list(logits))

            # Decode one symbol from the secret bits using the LLM distribution
            symbol_index = decoder.decode(table)
            token_id = indices[symbol_index]

            token_ids.append(token_id)
            generated.append(token_id)

        except Exception as e:
            print(f"Error during hiding: {e}")
            break

    return generated


def extract_bits(backend: TransformersBackend, prompt: str, cover_tokens: list[int]) -> CodedBits:
    """Extract bits from generated text (Receiver)."""
    token_ids = backend.tokenize(prompt)
    encoder = RangeEncoder()

    print(f"Extracting from {len(cover_tokens)} tokens...")

    for token_id in cover_tokens:
        logits = backend.next_logits(token_ids)
        indices, table = get_frequencies(list(logits))

        if token_id not in indices:
            raise RuntimeError("Cover token not in top-k! Cannot extract.")

        symbol_index = indices.index(token_id)
        encoder.encode(table, symbol_index)
        token_ids.append(token_id)

    return encoder.finish()


def main():  # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    # 1. Setup
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(manifest, cache_dir=DEFAULT_CACHE, local_files_only=True)

    secret_text = "極秘：今夜の会議は中止、明日10時にカフェで。"  # noqa: RUF001
    print(f"Secret: {secret_text}")

    # 2. Mock Secret LLM Compression with zlib for the spike
    compressed = zlib.compress(secret_text.encode("utf-8"), level=9)
    bits = CodedBits(compressed, len(compressed) * 8)
    print(f"Compressed size: {bits.bit_length} bits")

    # 3. Hide
    prompt = "架空のニュース記事：\n本日午後、東京都内の"  # noqa: RUF001
    cover_tokens = hide_bits(backend, prompt, bits)
    cover_text = backend.detokenize(cover_tokens)
    print(f"\n--- Cover Text ---\n{prompt}{cover_text}\n------------------\n")

    # 4. Extract
    extracted_bits = extract_bits(backend, prompt, cover_tokens)

    # The extracted bits might have extra 0s padded at the end due to RangeDecoder consuming more bits.  # noqa: E501
    # In real implementation we use a termination symbol or length header.
    # For spike, we just truncate to the original bytes.
    extracted_bytes = extracted_bits.data[: len(compressed)]

    try:
        recovered = zlib.decompress(extracted_bytes).decode("utf-8")
        print(f"Recovered: {recovered}")
        if recovered == secret_text:
            print("SUCCESS: Secret perfectly recovered!")
        else:
            print("FAILED: Mismatch.")
    except Exception as e:
        print(f"FAILED to decompress: {e}")


if __name__ == "__main__":
    sys.exit(main())  # type: ignore
