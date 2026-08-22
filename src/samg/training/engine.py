"""Compact train/validation engine with explicit checkpoint selection."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from samg.data import MuellerSplit, validate_disjoint_units
from samg.evaluation import evaluate_predictions
from samg.losses import (
    channel_scale,
    kl_divergence,
    normalized_matrix_mse,
    stokes_action_mse,
)
from samg.models import DirectMuellerVAE, FourIncidentPIVAE, OperatorPIVAE
from samg.physics import project_mueller_psd
from samg.training.schedules import PenaltySchedule, default_schedule


@dataclass
class TrainingConfig:
    model: str = "samg"
    latent_dim: int = 8
    epochs: int = 1000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    ridge: float = 1e-2
    seed: int = 2044
    device: str = "cuda"
    amp: str = "fp16"
    monitor_every: int = 25
    psd_numerical_tolerance: float = 1e-6
    psd_minimum_eigenvalue: float = 0.0
    compromise_physical_weight: float = 1.0
    cache_on_device: bool = False
    compile_model: bool = False

    def validate(self) -> TrainingConfig:
        if self.model not in {"samg", "samg-random", "operator", "direct"}:
            raise ValueError("model must be samg, samg-random, operator or direct")
        if self.epochs < 1 or self.batch_size < 1 or self.monitor_every < 1:
            raise ValueError("epochs, batch_size and monitor_every must be positive")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.weight_decay < 0 or self.ridge < 0:
            raise ValueError("weight_decay and ridge must be non-negative")
        if (
            self.gradient_clip <= 0
            or self.psd_numerical_tolerance < 0
            or self.psd_minimum_eigenvalue < 0
        ):
            raise ValueError("gradient_clip must be positive and tolerance non-negative")
        if self.amp not in {"off", "fp16", "bf16"}:
            raise ValueError("amp must be off, fp16 or bf16")
        return self


def build_model(name: str, *, latent_dim: int = 8) -> nn.Module:
    if name in {"samg", "samg-random"}:
        return FourIncidentPIVAE(latent_dim=latent_dim)
    if name == "operator":
        return OperatorPIVAE(latent_dim=latent_dim)
    if name == "direct":
        return DirectMuellerVAE(latent_dim=latent_dim, beta_kl=0.0)
    raise ValueError(f"Unknown model {name!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_line(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _generic_condition_penalty(states: torch.Tensor) -> torch.Tensor:
    condition = torch.nan_to_num(
        torch.linalg.cond(states.float()).clamp_min(1.0),
        nan=1e6,
        posinf=1e6,
        neginf=1e6,
    )
    return (condition - math.sqrt(3.0)).square().mean()


def _generic_poincare_penalty(states: torch.Tensor) -> torch.Tensor:
    intensity = states[..., 0].square().clamp_min(1e-8)
    polarization = states[..., 1:].square().sum(-1)
    return F.relu(1000.0 * polarization / intensity - 1000.0).mean()


def _penalty_per_sample(name: str, model_output: dict[str, torch.Tensor]) -> torch.Tensor:
    if name in {"cond_in", "cond_out"}:
        key = "s_in" if name == "cond_in" else "s_out"
        states = torch.nan_to_num(model_output[key].float(), nan=0.0, posinf=1e6, neginf=-1e6)
        condition = torch.nan_to_num(
            torch.linalg.cond(states).clamp_min(1.0),
            nan=1e6,
            posinf=1e6,
            neginf=1e6,
        )
        return (condition - math.sqrt(3.0)).square()
    if name == "poincare":
        states = model_output["s_out"].float()
        intensity = states[..., 0].square().clamp_min(1e-8)
        polarization = states[..., 1:].square().sum(-1)
        return F.relu(1000.0 * polarization / intensity - 1000.0).mean(-1)
    raise KeyError(name)


def _loss_terms(
    model: nn.Module,
    model_name: str,
    batch: dict[str, torch.Tensor],
    *,
    epoch: int,
    schedule: PenaltySchedule,
    scale: torch.Tensor,
    ridge: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    kwargs = {"sample_latent": True}
    if model_name != "direct":
        kwargs["pinv_ridge"] = ridge
    output = model(
        batch["patch"],
        batch["target"],
        batch["position"],
        batch["spectral"],
        **kwargs,
    )
    if epoch < schedule.switch_epoch and model_name != "direct":
        reconstruction = stokes_action_mse(output["s_out"], batch["target"], output["s_in"])
    elif epoch < schedule.switch_epoch:
        reconstruction = F.mse_loss(output["m_hat"], batch["target"])
    else:
        reconstruction = normalized_matrix_mse(output["m_hat"], batch["target"], scale)

    generator_kl = output.get("generator_kl", output["mu"].new_zeros(()))
    terms = {
        "reconstruction": reconstruction,
        "kl": kl_divergence(output["mu"], output["logvar"]) + generator_kl,
    }
    if model_name != "direct":
        terms.update(
            {
                "cond_in": _generic_condition_penalty(output["s_in"]),
                "cond_out": _generic_condition_penalty(output["s_out"]),
                "poincare": _generic_poincare_penalty(output["s_out"]),
            }
        )
    return terms, output


def _device_tensors(
    split: MuellerSplit, device: torch.device, cache: bool
) -> dict[str, torch.Tensor]:
    target = device if cache else torch.device("cpu")
    return split.tensors(target)


def _batch(
    tensors: dict[str, torch.Tensor], index: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    if next(iter(tensors.values())).device == device:
        return {name: value[index.to(device)] for name, value in tensors.items()}
    return {name: value[index].to(device, non_blocking=True) for name, value in tensors.items()}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    model_name: str,
    tensors: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    scale: torch.Tensor,
    ridge: float,
    psd_numerical_tolerance: float,
    psd_minimum_eigenvalue: float,
    include_psd_projection: bool = False,
) -> dict[str, float | list[float]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    emerging: list[torch.Tensor] = []
    for start in range(0, len(tensors["target"]), batch_size):
        index = torch.arange(start, min(start + batch_size, len(tensors["target"])))
        batch = _batch(tensors, index, device)
        kwargs = {"sample_latent": False}
        if model_name != "direct":
            kwargs["pinv_ridge"] = ridge
        output = model(
            batch["patch"],
            batch["target"],
            batch["position"],
            batch["spectral"],
            **kwargs,
        )
        predictions.append(output["m_hat"].float().cpu())
        targets.append(batch["target"].float().cpu())
        if "s_out" in output:
            emerging.append(output["s_out"].float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = evaluate_predictions(
        prediction,
        target,
        emerging=torch.cat(emerging) if emerging else None,
        numerical_tolerance=psd_numerical_tolerance,
        minimum_eigenvalue=psd_minimum_eigenvalue,
    )
    metrics["normalized_mse"] = float(normalized_matrix_mse(prediction, target, scale.cpu()))
    if include_psd_projection:
        projected = project_mueller_psd(prediction)
        projected_metrics = evaluate_predictions(
            projected,
            target,
            numerical_tolerance=psd_numerical_tolerance,
            minimum_eigenvalue=psd_minimum_eigenvalue,
        )
        for name, value in projected_metrics.items():
            metrics[f"posthoc_psd_{name}"] = value
        metrics["posthoc_psd_normalized_mse"] = float(
            normalized_matrix_mse(projected, target, scale.cpu())
        )
    return metrics


def evaluate_model_on_split(
    model: nn.Module,
    model_name: str,
    split: MuellerSplit,
    *,
    channel_scale_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    ridge: float = 1e-2,
    psd_numerical_tolerance: float = 1e-6,
    psd_minimum_eigenvalue: float = 0.0,
    include_psd_projection: bool = False,
) -> dict[str, float | list[float]]:
    """Evaluate a known architecture on one explicit split."""
    resolved = torch.device(device)
    model = model.to(resolved)
    tensors = _device_tensors(split, resolved, cache=False)
    return _evaluate(
        model,
        model_name,
        tensors,
        device=resolved,
        batch_size=batch_size,
        scale=channel_scale_tensor.to(resolved),
        ridge=ridge,
        psd_numerical_tolerance=psd_numerical_tolerance,
        psd_minimum_eigenvalue=psd_minimum_eigenvalue,
        include_psd_projection=include_psd_projection,
    )


def _learning_rate_factor(config: TrainingConfig, epoch: int) -> float:
    warmup = min(config.warmup_epochs, config.epochs)
    if epoch <= warmup:
        return epoch / max(1, warmup)
    progress = (epoch - warmup) / max(1, config.epochs - warmup)
    floor = config.min_learning_rate / config.learning_rate
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _gradient_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # compatibility with older supported PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_model(
    training: MuellerSplit,
    validation: MuellerSplit,
    output_directory: str | Path,
    config: TrainingConfig,
    *,
    schedule: PenaltySchedule | None = None,
) -> dict[str, object]:
    """Train without loading or evaluating any test split."""
    config.validate()
    if training.split != "train" or validation.split != "validation":
        raise ValueError("train_model requires explicit train and validation splits")
    validate_disjoint_units([training, validation])
    _seed_everything(config.seed)

    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(config.model, latent_dim=config.latent_dim).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    schedule = schedule or default_schedule(config.model)
    coefficient_rng = random.Random(config.seed)
    batch_rng = torch.Generator().manual_seed(config.seed)

    train_tensors = _device_tensors(training, device, config.cache_on_device)
    validation_tensors = _device_tensors(validation, device, config.cache_on_device)
    scale, degenerate = channel_scale(training.target)
    scale = scale.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    use_fp16 = device.type == "cuda" and config.amp == "fp16"
    scaler = _gradient_scaler(use_fp16)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[config.amp]
    if device.type != "cuda":
        amp_dtype = None

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if history_path.exists():
        raise FileExistsError(f"Refusing to append to an existing run: {history_path}")
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_observations": len(training),
        "validation_observations": len(validation),
        "train_units": len(set(training.unit_keys)),
        "validation_units": len(set(validation.unit_keys)),
        "constant_channels": [list(index) for index in degenerate.nonzero().tolist()],
        "selection": {
            "best_mse": "minimum validation normalized MSE",
            "best_physical": "maximum validation Cloude margin fraction; MSE tie-break",
            "best_compromise": "normalized MSE + weight * (1 - Cloude margin fraction)",
        },
        "schedule": {
            "switch_epoch": schedule.switch_epoch,
            "randomize_penalties": schedule.randomize,
            "calibration_window": schedule.calibration_window,
            "ramps": {
                name: {
                    "start": ramp.start,
                    "length": ramp.length,
                    "weight": ramp.weight,
                }
                for name, ramp in schedule.ramps.items()
            },
        },
        "test_opened": False,
        "torch": torch.__version__,
        "device": str(device),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best_mse = math.inf
    best_physical = (-math.inf, math.inf)
    best_compromise = math.inf
    best_epochs = {"mse": -1, "physical": -1, "compromise": -1}
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        factor = _learning_rate_factor(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * factor
        model.train()
        multipliers = schedule.resample(coefficient_rng)
        weights = schedule.weights(epoch)
        pending = schedule.pending(epoch)
        calibration_values: dict[str, list[float]] = {name: [] for name in pending}
        order = torch.randperm(len(training), generator=batch_rng)
        totals = {"loss": 0.0, "reconstruction": 0.0, "matrix_mse": 0.0}
        count = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = _batch(train_tensors, index, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                terms, model_output = _loss_terms(
                    model,
                    config.model,
                    batch,
                    epoch=epoch,
                    schedule=schedule,
                    scale=scale,
                    ridge=config.ridge,
                )
                loss = terms["reconstruction"]
                for name, weight in weights.items():
                    loss = loss + weight * terms[name]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            size = len(index)
            totals["loss"] += float(loss.detach()) * size
            totals["reconstruction"] += float(terms["reconstruction"].detach()) * size
            totals["matrix_mse"] += (
                float((model_output["m_hat"] - batch["target"]).square().mean().detach()) * size
            )
            count += size
            for name in pending:
                calibration_values[name].append(
                    float(_penalty_per_sample(name, model_output).median().detach())
                )

        schedule.observe(
            epoch,
            {
                name: float(np.median(values)) if values else 0.0
                for name, values in calibration_values.items()
            },
        )

        monitored = epoch == 1 or epoch == config.epochs or epoch % config.monitor_every == 0
        if not monitored:
            continue
        validation_metrics = _evaluate(
            model,
            config.model,
            validation_tensors,
            device=device,
            batch_size=config.batch_size,
            scale=scale,
            ridge=config.ridge,
            psd_numerical_tolerance=config.psd_numerical_tolerance,
            psd_minimum_eigenvalue=config.psd_minimum_eigenvalue,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "penalty_multipliers": multipliers,
            "penalty_weights": weights,
            "penalty_references": dict(schedule.references),
            "train_loss": totals["loss"] / count,
            "train_reconstruction": totals["reconstruction"] / count,
            "train_matrix_mse": totals["matrix_mse"] / count,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        _json_line(history_path, row)

        checkpoint = {
            "schema_version": 1,
            "model_name": config.model,
            "model_kwargs": {"latent_dim": config.latent_dim},
            "model_state": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation": validation_metrics,
            "channel_scale": scale.detach().cpu(),
            "config": asdict(config),
            "penalty_references": dict(schedule.references),
            "schedule": manifest["schedule"],
        }
        normalized_mse = float(validation_metrics["normalized_mse"])
        psd_fraction = float(validation_metrics["cloude_margin_fraction"])
        compromise = normalized_mse + config.compromise_physical_weight * (1.0 - psd_fraction)
        if normalized_mse < best_mse:
            best_mse = normalized_mse
            best_epochs["mse"] = epoch
            _atomic_torch_save(checkpoint, output / "best_mse.pt")
        if (psd_fraction, -normalized_mse) > (best_physical[0], -best_physical[1]):
            best_physical = (psd_fraction, normalized_mse)
            best_epochs["physical"] = epoch
            _atomic_torch_save(checkpoint, output / "best_physical.pt")
        if compromise < best_compromise:
            best_compromise = compromise
            best_epochs["compromise"] = epoch
            _atomic_torch_save(checkpoint, output / "best_compromise.pt")
        _atomic_torch_save(checkpoint, output / "last.pt")

    summary = {
        "best_epochs": best_epochs,
        "best_validation_normalized_mse": best_mse,
        "best_validation_cloude_margin_fraction": best_physical[0],
        "best_compromise_score": best_compromise,
        "elapsed_seconds": time.time() - started,
        "test_opened": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
