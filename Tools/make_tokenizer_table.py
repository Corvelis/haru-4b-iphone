#!/usr/bin/env python3
"""Pack the byte IDs and BPE merge rules from the pinned Haru tokenizer.json."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

# higurashi-ai/Haru-4B at 8d6e83bcd3701a76fc27a764965665da1df7da78.
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.tokenizer.read_bytes()
    if hashlib.sha256(source).hexdigest() != TOKENIZER_SHA256:
        parser.error("tokenizer.json does not match the pinned Haru-4B revision")

    model = json.loads(source)["model"]
    vocab = model["vocab"]
    output = bytearray(b"HBP1")
    extra = 256
    for byte in range(256):
        if 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255:
            symbol = chr(byte)
        else:
            symbol = chr(extra)
            extra += 1
        output.extend(struct.pack("<I", vocab[symbol]))

    for left, right in model["merges"]:
        output.extend(struct.pack("<III", vocab[left], vocab[right], vocab[left + right]))
    args.output.write_bytes(output)


if __name__ == "__main__":
    main()
