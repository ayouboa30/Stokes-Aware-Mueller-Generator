import json
from pathlib import Path

from samg.data import make_synthetic_split
from samg.training import TrainingConfig, default_schedule, train_model


def test_train_uses_only_train_and_validation_and_writes_all_checkpoints(tmp_path: Path):
    training = make_synthetic_split(16, split="train", seed=1, observations_per_unit=4)
    validation = make_synthetic_split(8, split="validation", seed=2, observations_per_unit=4)
    config = TrainingConfig(
        model="direct",
        epochs=1,
        batch_size=8,
        seed=4,
        device="cpu",
        amp="off",
        monitor_every=1,
    )
    schedule = default_schedule("direct", reconstruction_epochs=1, switch_epoch=1, ramp_length=0)
    summary = train_model(training, validation, tmp_path / "run", config, schedule=schedule)
    for name in ("best_mse.pt", "best_physical.pt", "best_compromise.pt", "last.pt"):
        assert (tmp_path / "run" / name).exists()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["test_opened"] is False
    assert summary["test_opened"] is False
    assert summary["best_validation_normalized_mse"] < 100.0
    assert "best_validation_cloude_margin_fraction" in summary
