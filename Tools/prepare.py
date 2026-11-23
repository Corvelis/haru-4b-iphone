#!/usr/bin/env python3
"""Prepare Haru's official weights on a Mac for the Haru iOS Metal runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

UPSTREAM = "7fe6b7665f8bf035998f401f8cb779fbe53f2751"
REVISION = "8d6e83bcd3701a76fc27a764965665da1df7da78"
MODEL = f"https://huggingface.co/higurashi-ai/Haru-4B/resolve/{REVISION}"
CHECKPOINT_BYTES = 10_540_113_510
TARGET_SHARD_BYTES = 192 * 1024**2


def japanese_vocabulary(tokenizer, required_ids):
    kana = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
    kanji = re.compile(r"[\u3400-\u9fff]{1,3}")
    ascii_word = re.compile(r"[\x20-\x7e]{1,5}")
    # Keep byte fallbacks and any prompt and duplex-control tokens required by the model.
    kept = set(range(256)) | set(required_ids)
    for token in range(tokenizer.get_vocab_size()):
        decoded = tokenizer.decode([token], skip_special_tokens=False).strip()
        if ("\ufffd" in decoded or kana.search(decoded) or
                kanji.fullmatch(decoded) or ascii_word.fullmatch(decoded)):
            kept.add(token)
    return sorted(kept)


def japanese_prompt(source, tokenizer):
    from haru_mlx.prompt import card_templates, render_card, system_prompt_ids

    card = json.loads((source / "haru_mlx/haru_mlx/assets/cards/ja.json").read_text())["chat"]
    card["situation"] = card["situation"].replace(
        "特定の話題や会話の流れは決めないが最初に自己紹介をしてから始める。", ""
    )
    return system_prompt_ids(tokenizer, render_card(card, card_templates()["templates"]["ja"]))


def quantization_bits(key, shape, args):
    if key.startswith("lm.layers.") and len(shape) == 2 and shape[-1] % 64 == 0:
        layer = int(key.split(".")[2])
        if args.profile == "japanese" and (
            layer < 4 or layer >= 24 or ("attn." in key and ".in_proj." not in key)
        ):
            return 8
        return args.bits
    if (key.startswith("codec.") and len(shape) == 2 and
            key.endswith((".fc1.weight", ".fc2.weight")) and shape[-1] >= 256):
        return args.encoder_bits if key.startswith("codec.encoder.") else args.decoder_bits
    if (key.startswith("head.") and key.endswith(".weight") and
            len(shape) == 2 and shape[-1] >= 256):
        return args.head_bits
    return 0


def download(url, destination, interval=None):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "4",
               "--connect-timeout", "15", "--speed-limit", "65536", "--speed-time", "45"]
    if interval:
        start, end = interval
        command += ["--range", f"{start}-{end - 1}", "--max-filesize", str(max(end - start, 65536))]
        url += f"?range={start}-{end - 1}"
    response = subprocess.check_output([*command, "-o", str(destination), "-w", "%{http_code}", url])
    if interval and (response != b"206" or destination.stat().st_size != end - start):
        raise ValueError(f"Hugging Face returned an invalid byte range for {interval}")


class Checkpoint:
    def __init__(self, work, local):
        self.local = local
        self.range_file = work / "range"
        self.offset = 0
        self.digest = hashlib.sha256()
        self.hashed = 0
        with self.read(0, 8) as file:
            self.offset = 8 + struct.unpack("<Q", file.read(8))[0]
        with self.read(8, self.offset) as file:
            self.tensors = json.loads(file.read(self.offset - 8))
        self.tensors.pop("__metadata__", None)

    def read(self, start, end):
        if self.local:
            file = self.local.open("rb")
            file.seek(start)
            return file
        download(f"{MODEL}/model.safetensors", self.range_file, (start, end))
        return self.range_file.open("rb")

    def batches(self):
        entries = sorted(self.tensors, key=lambda name: self.tensors[name]["data_offsets"][0])
        batch = []
        start = end = 0
        for name in entries:
            lo, hi = self.tensors[name]["data_offsets"]
            if batch and hi - start > TARGET_SHARD_BYTES:
                yield batch, start, end
                batch = []
            if not batch:
                start = lo
            batch.append(name)
            end = hi
        if batch:
            yield batch, start, end

    def arrays(self, names, start, end):
        import mlx.core as mx
        import numpy as np

        if not self.hashed:
            with self.read(0, self.offset) as file:
                header = file.read(self.offset)
            self.digest.update(header)
            self.hashed = len(header)
        # The embedding is much larger than other weights. Fetch it in pieces.
        if end - start > TARGET_SHARD_BYTES:
            (name,) = names
            shape = self.tensors[name]["shape"]
            row_bytes = shape[1] * 2
            rows_per_range = TARGET_SHARD_BYTES // row_bytes
            rows_per_range = min(rows_per_range, 16_384)
            ranges = [(first, min(rows_per_range, shape[0] - first))
                      for first in range(0, shape[0], rows_per_range)]
            def fetch(part):
                first, rows = part
                lo = self.offset + start + first * row_bytes
                size = rows * row_bytes
                if self.local:
                    with self.read(lo, lo + size) as file:
                        return file.read(size)
                path = self.range_file.with_name(f"embedding-{first}")
                try:
                    download(f"{MODEL}/model.safetensors", path, (lo, lo + size))
                    return path.read_bytes()
                finally:
                    path.unlink(missing_ok=True)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for index in range(0, len(ranges), 4):
                    batch = ranges[index:index + 4]
                    for (first, rows), data in zip(batch, pool.map(fetch, batch)):
                        self.hash_range(self.offset + start + first * row_bytes, data)
                        yield name, mx.array(np.frombuffer(data, dtype="<u2").reshape(rows, shape[1])).view(mx.bfloat16)
            return
        with self.read(self.offset + start, self.offset + end) as file:
            data = file.read(end - start)
        self.hash_range(self.offset + start, data)
        for name in names:
            entry = self.tensors[name]
            lo, hi = entry["data_offsets"]
            raw = np.frombuffer(data, dtype="<u2", count=(hi - lo) // 2, offset=lo - start)
            yield name, mx.array(raw.reshape(entry["shape"])).view(mx.bfloat16)

    def hash_range(self, start, data):
        if start != self.hashed:
            raise ValueError("Haru checkpoint has overlapping or missing bytes")
        self.digest.update(data)
        self.hashed += len(data)

    def verify(self):
        expected = "ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa"
        if self.hashed != CHECKPOINT_BYTES or self.digest.hexdigest() != expected:
            raise ValueError("Haru checkpoint SHA-256 does not match the official release")


def convert(checkpoint, config, output, args, vocabulary):
    import mlx.core as mx
    from haru_mlx import weights
    from haru_mlx.models.loaders import _convert

    plan = weights.conversion_plan({name: tuple(entry["shape"]) for name, entry in checkpoint.tensors.items()})
    weights.check_plan(plan, weights.parameter_shapes(config))
    target_for = {source: key for key, entry in plan.items() for source in entry.sources}
    waiting = {}
    shards = []
    overrides = {}
    embedding_offset = 0
    for names, start, end in checkpoint.batches():
        tensors = {}
        embedding_parts = []
        for source, value in checkpoint.arrays(names, start, end):
            key = target_for[source]
            entry = plan[key]
            is_embedding = key == "lm.embed_tokens.weight"
            if is_embedding:
                end_row = embedding_offset + value.shape[0]
                if vocabulary is not None:
                    rows = [token - embedding_offset for token in vocabulary if embedding_offset <= token < end_row]
                    value = mx.take(value, mx.array(rows, dtype=mx.int32), axis=0)
                embedding_offset = end_row
                if value.shape[0]:
                    parts = mx.quantize(value, group_size=64, bits=args.embedding_bits)
                    if args.embedding_bits != args.bits:
                        overrides[key] = args.embedding_bits
                    mx.eval(parts)
                    embedding_parts.append(parts)
                continue
            waiting[source] = value
            if any(name not in waiting for name in entry.sources):
                continue
            value = _convert(entry, waiting, None)
            for name in entry.sources:
                del waiting[name]
            quantization = quantization_bits(key, value.shape, args)
            if quantization and (quantization != args.bits or not key.startswith("lm.layers.")):
                overrides[key] = quantization
            if quantization:
                tensors[key], tensors[key + ".scales"], tensors[key + ".biases"] = mx.quantize(
                    value, group_size=64, bits=quantization)
            else:
                tensors[key] = mx.contiguous(value)
        if embedding_parts:
            key = "lm.embed_tokens.weight"
            for suffix, parts in zip(("", ".scales", ".biases"), zip(*embedding_parts)):
                tensors[key + suffix] = mx.concatenate(parts, axis=0)
            if vocabulary is not None:
                tensors["lm.embed_token_ids"] = mx.array(vocabulary, dtype=mx.int32)
        if tensors:
            filename = f"weights-{len(shards):03}.safetensors"
            mx.save_safetensors(str(output / filename), tensors)
            shards.append(filename)
        mx.clear_cache()
        print(f"重みを変換中 {end / (CHECKPOINT_BYTES - checkpoint.offset):.0%}", flush=True)
    if waiting:
        raise ValueError("Incomplete fused weights")
    checkpoint.verify()
    return shards, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("full-vocabulary", "japanese"), default="japanese",
                        help="The Japanese profile prioritizes 8-bit backbone layers and narrows the output vocabulary.")
    parser.add_argument("--bits", choices=(4, 5, 6, 8), type=int, default=6)
    parser.add_argument("--embedding-bits", choices=(4, 5, 6, 8), type=int,
                        help="Override the precision of the shared embedding and output projection.")
    parser.add_argument("--codec-bits", choices=(0, 8), type=int,
                        help="Override the precision of both codec directions. Use 0 to retain BF16.")
    parser.add_argument("--head-bits", choices=(0, 8), type=int, default=0,
                        help="Quantize the large MeanFlow head matrices. Defaults to BF16.")
    parser.add_argument("--checkpoint", type=Path, help="Existing Hugging Face checkpoint directory")
    parser.add_argument("--upstream", type=Path, help="Existing pinned Haru source directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    with tempfile.TemporaryDirectory(prefix="haru-", dir=output.parent) as temp:
        work = Path(temp)
        source = args.upstream
        if source is None:
            archive = work / "source.tar.gz"
            download(f"https://codeload.github.com/higurashi-ai/Haru/tar.gz/{UPSTREAM}", archive)
            with tarfile.open(archive) as file:
                # The pinned archive is public source. Extract only regular files needed by the converter.
                for member in file.getmembers():
                    relative = Path(member.name).parts[1:]
                    if member.isfile() and relative[:2] == ("haru_mlx", "haru_mlx") and ".." not in relative:
                        destination = work.joinpath(*relative)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with file.extractfile(member) as data, destination.open("wb") as target:
                            shutil.copyfileobj(data, target)
            source = work
        sys.path.insert(0, str(source.resolve() / "haru_mlx"))
        from haru_mlx.config import HaruConfig
        from tokenizers import Tokenizer

        for filename in ("config.json", "tokenizer.json"):
            if args.checkpoint:
                shutil.copyfile(args.checkpoint / filename, work / filename)
            else:
                download(f"{MODEL}/{filename}", work / filename)
        config_file = work / "config.json"
        config = HaruConfig.from_file(config_file)
        tokenizer = Tokenizer.from_file(str(work / "tokenizer.json"))
        prompt = japanese_prompt(source, tokenizer)
        vocabulary = None
        if args.profile == "japanese":
            vocabulary = japanese_vocabulary(tokenizer, [*prompt, config.pad_token_id, config.eop_token_id, config.eos_token_id])
        checkpoint = Checkpoint(work, args.checkpoint / "model.safetensors" if args.checkpoint else None)
        staging = work / "pack"
        staging.mkdir()
        args.embedding_bits = args.embedding_bits or (8 if vocabulary is not None else 6 if args.bits == 5 else args.bits)
        args.encoder_bits = 8 if args.codec_bits is None else args.codec_bits
        args.decoder_bits = 0 if args.codec_bits is None else args.codec_bits
        shards, overrides = convert(checkpoint, config, staging, args, vocabulary)
        shutil.copyfile(config_file, staging / "config.json")
        licenses = Path(__file__).resolve().parents[1] / "Sources" / "HaruKit" / "Licenses"
        shutil.copyfile(licenses / "Haru.txt", staging / "LICENSE-Haru")
        shutil.copyfile(licenses / "Haru-NOTICE.txt", staging / "NOTICE")
        manifest = {"format": 1, "bits": args.bits, "groupSize": 64, "precisionOverrides": overrides,
                    "prompt": prompt, "shards": shards, "revision": REVISION}
        if vocabulary is not None:
            manifest["embeddingTokenCount"] = len(vocabulary)
        (staging / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        print(output, flush=True)


if __name__ == "__main__":
    main()
