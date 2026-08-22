"""Sample a trained latent-field generator from a trusted checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from samg.generative import build_latent_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(arguments.checkpoint, map_location="cpu")
    model = build_latent_generator(payload["family"], **payload["model_kwargs"])
    model.load_state_dict(payload["model_state"], strict=True)
    device = torch.device(
        arguments.device
        if not arguments.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    normalized = model.sample(arguments.count, tuple(payload["spatial_shape"]), device, generator)
    fields = normalized.cpu() * torch.as_tensor(payload["latent_std"]) + torch.as_tensor(
        payload["latent_mean"]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "samg_generated_latent_fields",
            "family": payload["family"],
            "seed": arguments.seed,
            "fields": fields,
        },
        arguments.output,
    )
    print(f"wrote {len(fields)} latent fields to {arguments.output}")


if __name__ == "__main__":
    main()
