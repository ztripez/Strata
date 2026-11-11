"""Cross-check K2 expert slice decoding against pinned ggml C dequantizers."""

import argparse
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import gguf
import numpy as np


def check(source, common, slices, output):
    """Compile only the two existing ggml dequantizers and compare complete tensors."""
    source = Path(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    functions = []
    for name in ("iq3_s", "iq4_xs"):
        start = text.index("void dequantize_row_" + name + "(")
        pos = text.index("{", start)
        depth = 1
        end = pos + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        functions.append(text[start:end])
    code = """#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include <assert.h>
#include <string.h>
#define GGML_RESTRICT restrict
static float fp16(uint16_t bits) { _Float16 half; memcpy(&half, &bits, 2); return (float) half; }
#define GGML_FP16_TO_FP32 fp16
""" + "\n".join(functions)
    generated = out / "dequant_oracle.c"
    generated.write_text(code)
    library = out / "dequant_oracle.so"
    subprocess.run(
        [
            "cc",
            "-std=gnu11",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(Path(common).resolve()),
            str(generated),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library.resolve()))
    results = []
    for entry in json.loads((Path(slices) / "manifest.json").read_text()):
        if not entry["tensor"].startswith("blk.1."):
            continue
        path = Path(slices) / (entry["tensor"] + ".bin")
        raw = np.fromfile(path, dtype=np.uint8)
        kind = gguf.GGMLQuantizationType[entry["type"]]
        expected = gguf.dequantize(raw, kind)
        actual = np.empty_like(expected)
        fn = getattr(lib, "dequantize_row_" + entry["type"].lower())
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn(raw.ctypes.data, actual.ctypes.data, actual.size)
        np.testing.assert_array_equal(actual, expected)
        results.append(
            {
                "tensor": entry["tensor"],
                "type": entry["type"],
                "elements": actual.size,
                "bit_exact": bool(
                    np.array_equal(actual.view("u4"), expected.view("u4"))
                ),
            }
        )
    record = {
        "ggml_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "common_sha256": hashlib.sha256(
            (Path(common) / "ggml-common.h").read_bytes()
        ).hexdigest(),
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "common", "slices", "output"):
        parser.add_argument(name)
    a = parser.parse_args()
    check(a.source, a.common, a.slices, a.output)
