"""Probe the pinned model backend and emit bounded reproducibility evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from lsteg.model import ModelManifest, TransformersBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
DEFAULT_TEXT = "今日は研究室で再現可能な推論を確認する。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(
        manifest,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    token_ids = backend.tokenize(args.text)
    restored = backend.detokenize(token_ids)
    logits = backend.next_logits(token_ids)
    repeated = backend.next_logits(token_ids)
    if logits.sha256 != repeated.sha256:
        raise RuntimeError("repeated next-token logits differ in one process")
    token_bytes = b"".join(token_id.to_bytes(4, "big") for token_id in token_ids)

    evidence = {
        "schema_version": 1,
        "manifest": manifest.as_dict(),
        "runtime": backend.runtime.as_dict(),
        "input_sha256": sha256(args.text.encode("utf-8")).hexdigest(),
        "token_count": len(token_ids),
        "token_ids_sha256": sha256(token_bytes).hexdigest(),
        "tokenization_round_trip": restored == args.text,
        "vocabulary_size": backend.vocabulary_size,
        "logits_sha256": logits.sha256,
        "argmax_token_id": logits.argmax_token_id,
        "repeat_exact": True,
    }
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
