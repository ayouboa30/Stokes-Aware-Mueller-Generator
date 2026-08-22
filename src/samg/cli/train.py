"""Train SAMG or its direct VAE baseline using train and validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from samg.data import load_split
from samg.training import TrainingConfig, default_schedule, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("samg", "samg-random", "operator", "direct"), required=True
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--monitor-every", type=int, default=25)
    parser.add_argument("--psd-numerical-tolerance", type=float, default=1e-6)
    parser.add_argument("--psd-minimum-eigenvalue", type=float, default=0.0)
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--reconstruction-epochs", type=int, default=200)
    parser.add_argument("--switch-epoch", type=int, default=300)
    parser.add_argument("--ramp-length", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    training = load_split(arguments.train, expected_split="train")
    validation = load_split(arguments.validation, expected_split="validation")
    config = TrainingConfig(
        model=arguments.model,
        latent_dim=arguments.latent_dim,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        min_learning_rate=arguments.min_learning_rate,
        warmup_epochs=arguments.warmup_epochs,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        ridge=arguments.ridge,
        seed=arguments.seed,
        device=arguments.device,
        amp=arguments.amp,
        monitor_every=arguments.monitor_every,
        psd_numerical_tolerance=arguments.psd_numerical_tolerance,
        psd_minimum_eigenvalue=arguments.psd_minimum_eigenvalue,
        cache_on_device=arguments.cache_on_device,
        compile_model=arguments.compile_model,
    )
    schedule = default_schedule(
        arguments.model,
        reconstruction_epochs=arguments.reconstruction_epochs,
        switch_epoch=arguments.switch_epoch,
        ramp_length=arguments.ramp_length,
        kl_weight=arguments.kl_weight,
    )
    summary = train_model(training, validation, arguments.output, config, schedule=schedule)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
