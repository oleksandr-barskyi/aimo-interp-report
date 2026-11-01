#!/usr/bin/env python3
"""Download every public dataset of the aimo-interp HuggingFace org into data-hf/."""

from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-hf"
NAMES = (
    "augmented-sample-math",
    "augmented-sample-math-full-filtered",
    "augmented-sample-math-agg-filtered",
    "augmented-sample-math-agg",
    "aimo-interp-challenge-sample-full",
)
URL = "https://huggingface.co/datasets/aimo-interp/{}/resolve/main/data/validation-00000-of-00001.parquet"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in NAMES:
        target = OUT / f"{name}.parquet"
        with urlopen(URL.format(name), timeout=60) as stream:
            target.write_bytes(stream.read())
        print(f"{name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
