#!/usr/bin/env python3
"""Generate the iPhone installer plan from a pinned Safetensors header and tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

from prepare import (
    CHECKPOINT_BYTES, Checkpoint, REVISION, japanese_prompt, japanese_vocabulary,
    quantization_bits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path, help="The first 8 bytes and JSON header of model.safetensors")
    parser.add_argument("checkpoint_metadata", type=Path, help="Directory with config.json and tokenizer.json")
    parser.add_argument("upstream", type=Path, help="Pinned source directory for the Mac converter")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve() / "haru_mlx"))
    from haru_mlx import weights
    from haru_mlx.config import HaruConfig
    from tokenizers import Tokenizer

    header = args.header.read_bytes()
    assert len(header) == 8 + struct.unpack("<Q", header[:8])[0]
    entries = json.loads(header[8:])
    entries.pop("__metadata__", None)
    assert {v["dtype"] for v in entries.values()} == {"BF16"}
    assert max(v["data_offsets"][1] for v in entries.values()) + len(header) == CHECKPOINT_BYTES

    config = HaruConfig.from_file(args.checkpoint_metadata / "config.json")
    tokenizer = Tokenizer.from_file(str(args.checkpoint_metadata / "tokenizer.json"))
    prompt = japanese_prompt(args.upstream, tokenizer)
    vocabulary = japanese_vocabulary(
        tokenizer, [*prompt, config.pad_token_id, config.eop_token_id, config.eos_token_id],
    )
    preset = SimpleNamespace(profile="japanese", bits=6, encoder_bits=8, decoder_bits=0, head_bits=0)
    mappings = weights.conversion_plan({name: tuple(value["shape"]) for name, value in entries.items()})
    weights.check_plan(mappings, weights.parameter_shapes(config))
    targets = {source: name for name, mapping in mappings.items() for source in mapping.sources}
    checkpoint = Checkpoint.__new__(Checkpoint)
    checkpoint.tensors = entries
    batches, waiting, shards = [], set(), []
    embedding = "lm.embed_tokens.weight"
    overrides = {}
    for names, start, end in checkpoint.batches():
        inputs, outputs = [], []
        for name in names:
            item = entries[name]
            lo, hi = item["data_offsets"]
            inputs.append(dict(name=name, start=lo - start, end=hi - start, shape=item["shape"]))
            key = targets[name]
            mapping = mappings[key]
            waiting.add(name)
            if not all(part in waiting for part in mapping.sources):
                continue
            waiting.difference_update(mapping.sources)
            bits = 8 if key == embedding else quantization_bits(key, mapping.shape, preset)
            if bits and bits != preset.bits:
                overrides[key] = bits
            outputs.append(dict(name=key, sources=mapping.sources, layout=mapping.layout,
                                shape=mapping.shape, bits=bits))
        assert outputs
        shards.append(f"weights-{len(shards):03}.safetensors")
        batches.append(dict(start=start + len(header), end=end + len(header), inputs=inputs,
                            outputs=outputs, commit=not waiting))
    assert not waiting
    pack = dict(format=1, bits=preset.bits, groupSize=64, precisionOverrides=overrides, prompt=prompt,
                embeddingTokenCount=len(vocabulary), shards=shards, revision=REVISION)
    plan = dict(headerBytes=len(header), headerSHA256=hashlib.sha256(header).hexdigest(),
                embeddingIDs=vocabulary, batches=batches, pack=pack)
    args.output.write_text(json.dumps(plan, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
