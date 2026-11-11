"""Inspect a downloaded GGUF header slice without mapping tensor payloads."""

import argparse
import io
import json
import math
import struct
from pathlib import Path


def inspect(data):
    """Return metadata and tensor offset descriptors from a complete header slice."""
    stream = io.BytesIO(data)

    def number(fmt):
        size = struct.calcsize("<" + fmt)
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("header slice ends before tensor descriptors")
        return struct.unpack("<" + fmt, raw)[0]

    def string():
        size = number("Q")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("truncated string")
        return raw.decode("utf-8")

    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = number("I"), number("Q")
            return [value(subtype) for _ in range(count)]
        return number(formats[kind])

    if stream.read(4) != b"GGUF":
        raise ValueError("not GGUF")
    version, count, kv = number("I"), number("Q"), number("Q")
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")
    metadata = {}
    for _ in range(kv):
        key = string()
        metadata[key] = value(number("I"))
    tensors = []
    for _ in range(count):
        name = string()
        dims = [number("Q") for _ in range(number("I"))]
        kind, offset = number("I"), number("Q")
        tensors.append(
            {
                "name": name,
                "shape": dims,
                "type": kind,
                "offset": offset,
                "elements": math.prod(dims),
            }
        )
    alignment = metadata.get("general.alignment", 32)
    start = (stream.tell() + alignment - 1) // alignment * alignment
    for tensor in tensors:
        tensor["absolute_offset"] = start + tensor["offset"]
    return {
        "version": version,
        "data_start": start,
        "metadata": metadata,
        "tensors": tensors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.header.read_bytes())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "architecture",
        result["metadata"]["general.architecture"],
        "tensors",
        len(result["tensors"]),
        "data_start",
        result["data_start"],
    )
    for t in result["tensors"]:
        if t["name"].startswith(("blk.0.", "blk.1.")):
            print(t)
