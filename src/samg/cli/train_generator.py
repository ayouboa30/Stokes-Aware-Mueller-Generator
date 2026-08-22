"""Train one latent-field generator using train and validation only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from samg.data import load_latent_split, validate_latent_disjoint_units
from samg.generative import build_latent_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("pixelcnn", "pixelrnn", "vdm", "flow"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6, help="PixelCNN depth")
    parser.add_argument(
        "--steps", type=int, default=None, help="VDM or Flow Matching sampling steps"
    )
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monitor-every", type=int, default=10)
    return parser.parse_args()


def _model_kwargs(arguments: argparse.Namespace, channels: int) -> dict[str, int]:
    values = {"width": arguments.width, "channels": channels}
    if arguments.family == "pixelcnn":
        values["depth"] = arguments.depth
    if arguments.family in {"vdm", "flow"} and arguments.steps is not None:
        values["steps"] = arguments.steps
    return values


@torch.no_grad()
def _validation_loss(model, fields: torch.Tensor, batch_size: int, *, seed: int) -> float:
    model.eval()
    devices = [fields.device.index or 0] if fields.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if fields.is_cuda:
            torch.cuda.manual_seed(seed)
        total = 0.0
        for start in range(0, len(fields), batch_size):
            block = fields[start : start + batch_size]
            total += float(model.loss(block)) * len(block)
    return total / len(fields)


def main() -> None:
    arguments = parse_args()
    training = load_latent_split(arguments.train, expected_split="train")
    validation = load_latent_split(arguments.validation, expected_split="validation")
    validate_latent_disjoint_units(training, validation)
    if training.fields.shape[1:] != validation.fields.shape[1:]:
        raise SystemExit("Train and validation latent fields must have matching C,H,W")

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)
    device = torch.device(
        arguments.device
        if not arguments.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    channels = training.fields.shape[1]
    kwargs = _model_kwargs(arguments, channels)
    model = build_latent_generator(arguments.family, **kwargs).to(device)
    centre = training.fields.mean(dim=(0, 2, 3), keepdim=True)
    scale = training.fields.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    train_fields = ((training.fields - centre) / scale).to(device)
    validation_fields = ((validation.fields - centre) / scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    order_rng = torch.Generator(device=device).manual_seed(arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=True)
    history = arguments.output / "history.jsonl"
    if history.exists():
        raise FileExistsError(f"Refusing to append to {history}")
    manifest = {
        "family": arguments.family,
        "model_kwargs": kwargs,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "train_fields": len(training),
        "validation_fields": len(validation),
        "train_units": len(set(training.unit_keys)),
        "validation_units": len(set(validation.unit_keys)),
        "normalization": "per-channel mean and standard deviation from train only",
        "test_opened": False,
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    best = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        order = torch.randperm(len(train_fields), device=device, generator=order_rng)
        total = 0.0
        for start in range(0, len(order), arguments.batch_size):
            batch = train_fields[order[start : start + arguments.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite generator loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        if epoch not in {1, arguments.epochs} and epoch % arguments.monitor_every:
            continue
        validation_loss = _validation_loss(
            model, validation_fields, arguments.batch_size, seed=arguments.seed + 1_000_003
        )
        row = {
            "epoch": epoch,
            "train_loss": total / len(train_fields),
            "validation_loss": validation_loss,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        checkpoint = {
            "schema_version": 1,
            "family": arguments.family,
            "model_kwargs": kwargs,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "latent_mean": centre,
            "latent_std": scale,
            "spatial_shape": list(training.fields.shape[-2:]),
        }
        torch.save(checkpoint, arguments.output / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(checkpoint, arguments.output / "best.pt")
    print(json.dumps({"best_validation_loss": best, "test_opened": False}, indent=2))


if __name__ == "__main__":
    main()
