#!/usr/bin/env python3
"""Download the public aimo-interp datasets used by the report into data-hf/,
pinned to the HuggingFace revision the report was computed on, and check the
sha256 of every file."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-hf"
URL = "https://huggingface.co/datasets/aimo-interp/{}/resolve/{}/{}"
PINNED = {
    "aimo-interp-challenge-sample-full": (
        "c0ffb7e678294cc8819d50d979ba28ba49bd02d0",
        "data/validation-00000-of-00001.parquet",
        "2989ab6a018f2779a821209f414bd0049c9ef1c2488f589ff2d3bf4ed16942ff",
    ),
    "train-main-v2": (
        "fa8af758c6519fc97a9d0bd9017feb6f12232b52",
        "data/train-00000-of-00001.parquet",
        "e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285",
    ),
}
UNPINNED = (
    "augmented-sample-math",
    "augmented-sample-math-full-filtered",
    "augmented-sample-math-agg-filtered",
    "augmented-sample-math-agg",
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, (rev, path, sha) in PINNED.items():
        target = OUT / f"{name}.parquet"
        with urlopen(URL.format(name, rev, path), timeout=60) as stream:
            data = stream.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got} != pinned {sha}")
        target.write_bytes(data)
        print(f"{name}@{rev[:8]}: {len(data)} bytes, sha256 ok")
    for name in UNPINNED:
        target = OUT / f"{name}.parquet"
        with urlopen(URL.format(name, "main", "data/validation-00000-of-00001.parquet"), timeout=60) as stream:
            target.write_bytes(stream.read())
        print(f"{name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
