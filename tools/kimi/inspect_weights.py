"""Read a safetensors header over HTTP without downloading its tensor payload."""

import argparse
import json
import struct
import urllib.request


def read_header(url):
    """Fetch the length and JSON header using bounded HTTP range requests."""

    def fetch(start, end):
        request = urllib.request.Request(
            url + ("&" if "?" in url else "?") + f"range={start}-{end}",
            headers={"Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError(
                    f"server ignored range request: HTTP {response.status}"
                )
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError("unexpected range response length")
        return data

    size = struct.unpack("<Q", fetch(0, 7))[0]
    if size > 100_000_000:
        raise ValueError(f"header too large: {size}")
    return json.loads(fetch(8, 7 + size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    header = read_header(args.url)
    with open(args.config) as stream:
        config = json.load(stream)["text_config"]["linear_attn_config"]
    name = "language_model.model.layers.0.self_attn.A_log"
    actual = header[name]["shape"]
    expected = [config["num_heads"]]
    print(f"{name}: checkpoint={actual}, expected={expected}")
    if actual != expected:
        print(
            "A_log has storage padding; layer0.Weights forward checks the unused values are zero."
        )
