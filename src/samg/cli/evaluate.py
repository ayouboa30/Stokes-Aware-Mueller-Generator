"""Evaluate one explicit split from a known SAMG checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from samg.data import load_split
from samg.training import build_model, evaluate_model_on_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("samg", "samg-random", "operator", "direct"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--psd-numerical-tolerance", type=float, default=1e-6)
    parser.add_argument("--psd-minimum-eigenvalue", type=float, default=0.0)
    parser.add_argument("--include-psd-projection", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model != arguments.model:
        raise SystemExit(
            f"Checkpoint architecture is {checkpoint_model!r}, not {arguments.model!r}"
        )
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = build_model(arguments.model, latent_dim=int(model_kwargs.get("latent_dim", 8)))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    split = load_split(arguments.data)
    device = arguments.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    metrics = evaluate_model_on_split(
        model,
        arguments.model,
        split,
        channel_scale_tensor=torch.as_tensor(checkpoint["channel_scale"]),
        device=device,
        batch_size=arguments.batch_size,
        ridge=arguments.ridge,
        psd_numerical_tolerance=arguments.psd_numerical_tolerance,
        psd_minimum_eigenvalue=arguments.psd_minimum_eigenvalue,
        include_psd_projection=arguments.include_psd_projection,
    )
    payload = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model": arguments.model,
        "split": split.split,
        "observations": len(split),
        "units": len(set(split.unit_keys)),
        "test_evaluated": split.split == "test",
        "protocol": {
            "evaluation_batch_size": arguments.batch_size,
            "training_batch_size": checkpoint.get("config", {}).get("batch_size"),
            "ridge": arguments.ridge,
            "psd_numerical_tolerance": arguments.psd_numerical_tolerance,
            "psd_minimum_eigenvalue": arguments.psd_minimum_eigenvalue,
            "posthoc_psd_projection": arguments.include_psd_projection,
            "historical_grouping": (
                "four contiguous minibatch groups"
                if arguments.model in {"samg", "samg-random"}
                else None
            ),
        },
        "metrics": metrics,
    }
    training_batch = payload["protocol"]["training_batch_size"]
    if arguments.model in {"samg", "samg-random"} and training_batch != arguments.batch_size:
        payload["warning"] = (
            "The historical SAMG incident banks depend on minibatch grouping; "
            "evaluation batch size differs from training."
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
