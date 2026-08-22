"""Convert one explicitly trusted legacy checkpoint to a tensor-only payload.

Legacy pickle loading is intentionally gated because an untrusted checkpoint
can execute arbitrary code. This tool should be used once, offline, after
verifying the source and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a mapping")
    candidate = payload
    for key in ("model_state", "model", "state_dict"):
        if key in payload:
            candidate = payload[key]
            break
    if not isinstance(candidate, dict) or not candidate:
        raise ValueError("No non-empty state dictionary found")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items()):
        raise ValueError("State dictionary must contain string keys and tensors only")
    return {key: value.detach().cpu() for key, value in candidate.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--i-understand-checkpoint-is-trusted",
        action="store_true",
        help="required acknowledgement before legacy pickle loading",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if not arguments.i_understand_checkpoint_is_trusted:
        raise SystemExit("Refusing legacy load without the explicit trust acknowledgement")
    payload = torch.load(arguments.input, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "model_state": state}, arguments.output)
    report = {
        "input_sha256": file_sha256(arguments.input),
        "output_sha256": file_sha256(arguments.output),
        "tensor_count": len(state),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
