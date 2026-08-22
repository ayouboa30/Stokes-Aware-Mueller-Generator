"""Create deterministic synthetic splits for smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from samg.data import make_synthetic_split, save_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2044)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for offset, (name, size) in enumerate(
        [
            ("train", arguments.train_size),
            ("validation", arguments.validation_size),
            ("test", arguments.test_size),
        ]
    ):
        split = make_synthetic_split(size, split=name, seed=arguments.seed + offset)
        save_split(split, arguments.output / f"{name}.pt")
        print(f"wrote {name}: {len(split)} observations")


if __name__ == "__main__":
    main()
